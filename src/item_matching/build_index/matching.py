import sys
from pathlib import Path
import polars as pl
from loguru import logger
from .func import tfidf, make_dir
from .model import Model

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


class Data:
    def __init__(
            self,
            path: Path,
            shard_size: int = 1_500_000
    ):
        self.path = path
        self.shard_size = shard_size

    def create_dataset(
            self,
            data: pl.DataFrame,
            pp,
            fn_kwargs: dict,
            col_embed: str,
            mode: str = '',
            batch_size: int = 512,
            pp_proc=None,
            fn_kwargs_proc: dict = None,
    ) -> dict[str, Path]:
        from datasets import Dataset
        import numpy as np

        # input
        path_tmp = {}
        for i in [f'{mode}_array', f'{mode}_ds']:
            path_tmp.update({i: self.path / i})
            make_dir(path_tmp[i])

        # batch save embeddings
        total_sample = len(data)
        num_chunks = total_sample // self.shard_size + 1
        logger.info(f'[Dataset] Total chunks {mode}: {num_chunks} - Shard size: {self.shard_size}')

        for idx, b in enumerate(range(0, total_sample, self.shard_size)):
            # chunking
            if b + self.shard_size >= total_sample:
                dataset = Dataset.from_pandas(data[b:].to_pandas())
            else:
                dataset = Dataset.from_pandas(data[b:b + self.shard_size].to_pandas())
            # path
            path_tmp_array = path_tmp[f'{mode}_array'] / f'{idx}.npy'
            path_tmp_ds = path_tmp[f'{mode}_ds'] / f'{idx}'
            # process
            if pp_proc:
                dataset = dataset.map(pp_proc, batched=True, batch_size=2048, fn_kwargs=fn_kwargs_proc)
            dataset = dataset.map(pp, batched=True, batch_size=batch_size, fn_kwargs=fn_kwargs)
            dataset.set_format(type='numpy', columns=[col_embed], output_all_columns=True)
            # save chunking
            np.save(path_tmp_array, dataset[col_embed])
            dataset.save_to_disk(path_tmp_ds)

        return path_tmp


class BELargeScale:
    def __init__(
            self,
            path: Path,
            text_sparse: int = None,
            img_dim: bool = False,
            text_dense: bool = False,
            query_batch_size: int = 500_000
    ):
        self.text_sparse = text_sparse
        self.img_dim = img_dim
        self.text_dense = text_dense
        self.path = path
        self.data = Data(self.path)
        self.model = Model()
        self.query_batch_size = query_batch_size

    def transform_tfidf(
            self,
            df_db: pl.DataFrame,
            df_q: pl.DataFrame,
            col_embed: str
    ) -> dict:
        # tf-idf
        all_items = list(set(df_db['db_item_name_clean'].to_list() + df_q['q_item_name_clean'].to_list()))
        vectorizer = tfidf(all_items, dim=self.text_sparse)
        # transform embed
        dict_tmp = {}
        for mode, data in zip(['db', 'q'], [df_db, df_q]):
            fn_kwargs = {'col': f'{mode}_item_name_clean', 'vectorizer': vectorizer}
            path_tmp = self.data.create_dataset(
                data, mode=mode, pp=self.model.pp_sparse_tfidf, fn_kwargs=fn_kwargs,
                col_embed=col_embed, batch_size=768
            )
            dict_tmp[mode] = path_tmp
        return dict_tmp

    def transform_img(
            self,
            df_db: pl.DataFrame,
            df_q: pl.DataFrame,
            col_embed: str
    ) -> dict:
        # load model
        img_model, img_processor = self.model.get_img_model(model_id='openai/clip-vit-base-patch32')
        # transform embed
        dict_tmp = {}
        for mode, data in zip(['db', 'q'], [df_db, df_q]):
            # batch embed
            fn_kwargs = {'col': f'{mode}_file_path', 'model': img_model, 'processor': img_processor}
            path_tmp = self.data.create_dataset(
                data, mode=mode, pp=self.model.pp_img, fn_kwargs=fn_kwargs, col_embed=col_embed
            )
            dict_tmp[mode] = path_tmp
        return dict_tmp

    def transform_text_dense(
            self,
            df_db: pl.DataFrame,
            df_q: pl.DataFrame,
            col_embed: str
    ) -> dict:
        # model
        text_model = self.model.get_text_model()
        # transform embed
        dict_tmp = {}
        for mode, data in zip(['db', 'q'], [df_db, df_q]):
            fn_kwargs = {'col': f'{mode}_item_name_clean', 'model': text_model}
            path_tmp = self.data.create_dataset(
                data, mode=mode, pp=self.model.pp_dense, fn_kwargs=fn_kwargs, col_embed=col_embed
            )
            dict_tmp[mode] = path_tmp
        return dict_tmp

    def match(self, df_db: pl.DataFrame, df_q: pl.DataFrame, top_k: int = 10):
        from autofaiss import build_index
        from datasets import concatenate_datasets, load_from_disk
        from time import perf_counter

        # Dataset
        path_tmp, col_embed = None, None
        if self.text_sparse:
            # embed col
            col_embed = 'tfidf_embed'
            path_tmp = self.transform_tfidf(df_db, df_q, col_embed)

        elif self.img_dim:
            # embed col
            col_embed = 'img_embed'
            path_tmp = self.transform_img(df_db, df_q, col_embed)

        elif self.text_dense:
            col_embed = 'dense_embed'
            path_tmp = self.transform_text_dense(df_db, df_q, col_embed)

        # Build index
        logger.info(f'[Matching] Start building index')
        start = perf_counter()
        path_index = self.path / 'index'
        build_index(
            str(path_tmp['db']['db_array']),
            index_path=str(path_index / f'ip.index'),
            index_infos_path=str(path_index / f'index.json'),
            save_on_disk=True,
            metric_type='ip',
            verbose=30,
        )
        logger.info(f'[Matching] Building Index: {perf_counter() - start:,.2f}s')

        # Load dataset shard
        dataset_db = concatenate_datasets([
            load_from_disk(str(f)) for f in sorted(path_tmp['db']['db_ds'].glob('*'))
        ])

        # Add index
        dataset_db.load_faiss_index(col_embed, path_index / f'ip.index')

        # Dataset query shard
        dataset_q = concatenate_datasets([
            load_from_disk(str(f)) for f in sorted(path_tmp['q']['q_ds'].glob('*'))
        ])

        # Batch query
        self.path_result = self.path / 'result'
        make_dir(self.path_result)

        num_batches = len(dataset_q) // self.query_batch_size
        logger.info(f'[Matching] Start retrieve: num batches {num_batches}')
        start = perf_counter()
        for idx, i in enumerate(range(0, len(dataset_q), self.query_batch_size)):
            if i + self.query_batch_size >= len(dataset_q):
                batched_queries = dataset_q[i:]
            else:
                batched_queries = dataset_q[i:i + self.query_batch_size]

            # query
            score, result = dataset_db.get_nearest_examples_batch(
                col_embed,
                batched_queries[col_embed],
                k=top_k
            )

            dict_ = {f'score_{col_embed}': [list(i) for i in score]}
            df_score = pl.DataFrame(dict_)
            df_score.write_parquet(self.path_result / f'score_{idx}.parquet')

            df_result = pl.DataFrame(result).drop([col_embed])
            print(f'[Matching] Batch {idx}/{num_batches} match result shape: {df_result.shape}')
            df_result.write_parquet(self.path_result / f'result_{idx}.parquet')

            del score, result, df_score, df_result
        logger.info(f'[Matching] Retrieve: {perf_counter() - start:,.2f}s')

        # Post process
        df_score = (
            pl.concat([pl.read_parquet(f) for f in sorted(self.path_result.glob('score*.parquet'))])
        )
        df_result = (
            pl.concat([pl.read_parquet(f) for f in sorted(self.path_result.glob('result*.parquet'))])
        )
        df_match = pl.concat([df_q, df_result, df_score], how='horizontal')
        col_explode = [i for i in df_match.columns if 'db' in i] + [f'score_{col_embed}']
        df_match = df_match.explode(col_explode)

        return df_match

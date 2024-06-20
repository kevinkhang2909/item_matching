from pathlib import Path
import numpy as np
import polars as pl
from time import perf_counter
from re import search
from autofaiss import build_index
from datasets import concatenate_datasets, load_from_disk, Dataset
from item_matching.func.utilities import make_dir
from item_matching.model.model import Model
import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


class Data:
    def __init__(self, path: Path, shard_size: int = 1_500_000):
        self.path = path
        self.shard_size = shard_size

    def create_dataset(self, data: pl.DataFrame, pp, fn_kwargs: dict, col_embed: str, mode: str = '', batch_size: int = 256) -> dict[str, Path]:
        # Create directories
        path_array = self.path / f'{mode}_array'
        path_ds = self.path / f'{mode}_ds'
        make_dir(path_array)
        make_dir(path_ds)

        # Initialize paths
        path_tmp = {f'{mode}_array': path_array, f'{mode}_ds': path_ds}

        # Log total chunks
        total_sample = len(data)
        num_chunks = (total_sample + self.shard_size - 1) // self.shard_size
        logger.info(f'[Dataset] Total chunks {mode}: {num_chunks} - Shard size: {self.shard_size}')

        # Process and save each chunk
        for idx in range(num_chunks):
            start_idx = idx * self.shard_size
            end_idx = min(start_idx + self.shard_size, total_sample)
            # dataset_chunk = Dataset.from_pandas(data[start_idx:end_idx].to_pandas())
            dataset_chunk = Dataset.from_polars(data[start_idx:end_idx])

            # Process dataset
            dataset_chunk = dataset_chunk.map(pp, batched=True, batch_size=batch_size, fn_kwargs=fn_kwargs)
            dataset_chunk.set_format(type='numpy', columns=[col_embed], output_all_columns=True)

            # Save chunk
            np.save(path_array / f'{idx}.npy', dataset_chunk[col_embed])
            dataset_chunk.save_to_disk(str(path_ds / f'{idx}'))

        return path_tmp


class BELargeScale:
    def __init__(
            self,
            path: Path,
            img_dim: bool = False,
            text_dense: bool = False,
            query_batch_size: int = 500_000
    ):
        self.img_dim = img_dim
        self.text_dense = text_dense
        self.path = path
        self.data = Data(self.path)
        self.model = Model()
        self.query_batch_size = query_batch_size

    def transform_img(
            self,
            df_db: pl.DataFrame,
            df_q: pl.DataFrame,
            col_embed: str
    ) -> dict:
        # load model
        img_model, img_processor = self.model.get_img_model()
        # transform embed
        dict_tmp = {}
        for mode, data in zip(['db', 'q'], [df_db, df_q]):
            # batch embed
            fn_kwargs = {'col': f'{mode}_file_path', 'model': img_model, 'processor': img_processor}
            path_tmp = self.data.create_dataset(
                data, mode=mode, pp=self.model.pp_img, fn_kwargs=fn_kwargs, col_embed=col_embed, batch_size=128
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
        # Dataset
        col_embed = 'img_embed' if self.img_dim else 'dense_embed'
        path_tmp = {
            'db': {'db_array': self.path / 'db_array', 'db_ds': self.path / 'db_ds'},
            'q': {'q_array': self.path / 'q_array', 'q_ds': self.path / 'q_ds'},
        }
        transform_dict = {
            'img_embed': self.transform_img,
            'dense_embed': self.transform_text_dense,
        }
        if not path_tmp['db'][f'db_array'].exists():
            path_tmp = transform_dict[col_embed](df_db, df_q, col_embed)
        else:
            logger.info(f'[Matching] Dataset is existed')

        # Build index
        logger.info(f'[Matching] Start building index')
        start = perf_counter()
        path_index = self.path / 'index'
        if not path_index.exists():
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
            # init
            file_name_result = self.path_result / f'result_{idx}.parquet'
            file_name_score = self.path_result / f'score_{idx}.parquet'
            if file_name_result.exists():
                continue

            # start
            start_batch = perf_counter()
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
            # export
            dict_ = {f'score_{col_embed}': [list(i) for i in score]}
            df_score = pl.DataFrame(dict_)
            df_score.write_parquet(file_name_score)
            df_result = pl.DataFrame(result).drop([col_embed])
            df_result.write_parquet(file_name_result)

            # log
            print(f"[Matching] Batch {idx}/{num_batches} match result shape: {df_result.shape} "
                  f"{perf_counter() - start_batch:,.2f}s")
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
        col_explode = [i for i in df_match.columns if search('db|score', i)]
        df_match = df_match.explode(col_explode)

        return df_match

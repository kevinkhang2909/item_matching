import sys
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
import numpy as np
import polars as pl
from autofaiss import build_index
from datasets import Dataset, concatenate_datasets, load_from_disk
from loguru import logger
from func import tfidf, pp_text, make_dir

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}</cyan> | <level>{message}</level>')


class Data:
    def __init__(self, path: Path, batch_size: int = 1_500_000):
        self.path = path
        self.batch_size = batch_size

    def create_dataset(self, data, preprocess, pp, mode: str = ''):
        # input
        path_tmp = {}
        for i in [f'{mode}_array', f'{mode}_ds']:
            path_tmp.update({i: self.path / i})
            make_dir(path_tmp[i])

        # model
        fn_kwargs = {'col': f'{mode}_item_name_clean', 'preprocess': preprocess}

        # save embeddings into arrays
        total_sample = len(data)
        num_chunks = total_sample // self.batch_size
        logger.info(f'[Dataset] Total chunks {mode}: {num_chunks}')

        for idx, b in tqdm(enumerate(range(0, total_sample, self.batch_size)), total=num_chunks):
            # chunking
            if i + self.batch_size >= total_sample:
                dataset = Dataset.from_pandas(data[b:].to_pandas())
            else:
                dataset = Dataset.from_pandas(data[b:b + self.batch_size].to_pandas())
            # path
            path_tmp_array = path_tmp[f'{mode}_array'] / f'{idx}.npy'
            path_tmp_ds = path_tmp[f'{mode}_ds'] / f'{idx}'
            # process
            dataset = dataset.map(pp, batched=True, batch_size=512, fn_kwargs=fn_kwargs)
            dataset.set_format(type='numpy', columns=['embeddings'], output_all_columns=True)
            # save chunking
            np.save(path_tmp_array, dataset['embeddings'])
            dataset.save_to_disk(path_tmp_ds)

        return path_tmp


class BELargeScale:
    def __init__(
            self,
            path: Path,
            text_sparse: int = 512,
            img_dim: int = None,
            text_dense: int = None,
    ):
        self.text_sparse = text_sparse
        self.path = path

    def match(self, df_db: pl.DataFrame, df_q: pl.DataFrame, top_k: int = 10):
        # tf-idf
        all_items = list(set(df_db['db_item_name_clean'].to_list() + df_q['q_item_name_clean'].to_list()))
        vectorizer = tfidf(all_items, dim=self.text_sparse)

        # dataset
        data = Data(self.path)
        path_tmp_db, path_tmp_q = None, None
        if self.text_sparse:
            path_tmp_db = data.create_dataset(df_db, mode='db', preprocess=vectorizer, pp=pp_text)
            path_tmp_q = data.create_dataset(df_q, mode='q', preprocess=vectorizer, pp=pp_text)
            del df_db

        # build index
        start = perf_counter()
        path_index = self.path / 'index'
        build_index(
            str(path_tmp_db['db_array']),
            index_path=str(path_index / f'ip.index'),
            index_infos_path=str(path_index / f'index.json'),
            save_on_disk=True,
            metric_type='ip',
            verbose=30,
        )
        logger.info(f'[Matching] Index: {perf_counter() - start:,.2f}s')

        # load dataset shard
        dataset_db = concatenate_datasets([
            load_from_disk(str(f)) for f in sorted(path_tmp_db['db_ds'].glob('*'))
        ])

        # add index
        dataset_db.load_faiss_index('embeddings', path_index / f'ip.index')

        # dataset query shard
        dataset_q = concatenate_datasets([
            load_from_disk(str(f)) for f in sorted(path_tmp_q['q_ds'].glob('*'))
        ])

        # batch query
        batch_size = 500_000
        path_result = self.path / 'result'
        make_dir(path_result)

        start = perf_counter()
        for idx, i in tqdm(enumerate(range(0, len(dataset_q), batch_size)), total=len(dataset_q) // batch_size):
            if i + batch_size >= len(dataset_q):
                batched_queries = dataset_q[i:]
            else:
                batched_queries = dataset_q[i:i + batch_size]

            # query
            batched_query_embeddings = np.asarray(batched_queries['embeddings'])
            score, result = dataset_db.get_nearest_examples_batch(
                'embeddings',
                batched_query_embeddings,
                k=top_k
            )

            dict_ = {'score': [list(i) for i in score]}
            df_score = pl.DataFrame(dict_)
            df_score.write_parquet(path_result / f'score_{idx}.parquet')

            df_result = pl.DataFrame(result).drop(['embeddings'])
            df_result.write_parquet(path_result / f'result_{idx}.parquet')
            del score, result, df_score, df_result

        logger.info(f'[Matching] Query: {perf_counter() - start:,.2f}s')

        # post process
        df_score = (
            pl.concat([pl.read_parquet(f) for f in sorted(path_result.glob('score*.parquet'))])
        )
        df_result = (
            pl.concat([pl.read_parquet(f) for f in sorted(path_result.glob('result*.parquet'))])
        )
        df_match = pl.concat([df_q, df_result, df_score], how='horizontal')
        col_explode = [i for i in df_match.columns if 'db' in i] + ['score']
        df_match = df_match.explode(col_explode)

        return df_match

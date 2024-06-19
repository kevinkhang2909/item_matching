from pathlib import Path
import polars as pl
from time import perf_counter
from re import search
from autofaiss import build_index
from datasets import concatenate_datasets, load_from_disk
from pydantic import BaseModel, Field, computed_field
from item_matching.func.utilities import make_dir
from loguru import logger


class ConfigQuery(BaseModel):
    ROOT_PATH: Path = Field(default=None)
    QUERY_SIZE: int = Field(default=50_000)
    MODE: str = Field(default='')
    MATCH_BY: str = Field(default='text')
    TOP_K: int = Field(default=10)

    @computed_field
    @property
    def col_embedding(self) -> str:
        dict_ = {'image': 'image_embed'}
        return dict_.get(self.MATCH_BY, 'dense_embed')

    @computed_field
    @property
    def path_array_db(self) -> Path:
        return self.ROOT_PATH / f'db_array'

    @computed_field
    @property
    def path_ds_db(self) -> Path:
        return self.ROOT_PATH / f'db_ds'

    @computed_field
    @property
    def path_ds_q(self) -> Path:
        return self.ROOT_PATH / f'q_ds'

    @computed_field
    @property
    def path_index(self) -> Path:
        return self.ROOT_PATH / f'index'

    @computed_field
    @property
    def path_result(self) -> Path:
        path_result = self.ROOT_PATH / f'result'
        make_dir(path_result)
        return path_result


class BuildIndexAndQuery:
    def __init__(self, config: ConfigQuery):
        self.config = config
        self.file_index = None
        self.dataset_db = None
        self.dataset_q = None
        self.df_q = None

    def build(self):
        # Build index
        logger.info(f'[BuildIndex] Start building index')
        start = perf_counter()
        path_index = self.config.path_index
        self.file_index = str(path_index / f'ip.index')
        if not path_index.exists():
            build_index(
                str(self.config.path_array_db),
                index_path=self.file_index,
                index_infos_path=str(path_index / f'index.json'),
                save_on_disk=True,
                metric_type='ip',
                verbose=30,
            )
            logger.info(f'Building Index: {perf_counter() - start:,.2f}s')
        else:
            logger.info(f'Index is existed')

    def load_dataset_index(self):
        # Load dataset shard
        self.dataset_db = concatenate_datasets([
            load_from_disk(str(f)) for f in sorted(self.config.path_ds_db.glob('*'))
        ])

        # Add index
        self.dataset_db.load_faiss_index(self.config.col_embedding, self.file_index)

        # Dataset query shard
        self.dataset_q = concatenate_datasets([
            load_from_disk(str(f)) for f in sorted(self.config.path_ds_q.glob('*'))
        ])
        logger.info(f'[Query] Shard Loaded')

    def query(self, df_q: pl.DataFrame):
        # Load
        self.load_dataset_index()

        # Batch query
        total_sample = len(self.dataset_q)
        num_batches = (total_sample + self.config.QUERY_SIZE) // self.config.QUERY_SIZE
        logger.info(f'Total batches: {num_batches} - Query size: {self.config.QUERY_SIZE:,.0f}')
        for i, idx in enumerate(range(num_batches), start=1):
            # init
            file_name_result = self.config.path_result / f'result_{idx}.parquet'
            file_name_score = self.config.path_result / f'score_{idx}.parquet'
            if file_name_result.exists():
                continue

            # start
            start_batch = perf_counter()
            start_idx = idx * self.config.QUERY_SIZE
            end_idx = min(start_idx + self.config.QUERY_SIZE, total_sample)
            batched_queries = self.dataset_q[start_idx:end_idx]

            # query
            score, result = self.dataset_db.get_nearest_examples_batch(
                self.config.col_embedding,
                batched_queries[self.config.col_embedding],
                k=self.config.TOP_K,
            )
            # export
            dict_ = {f'score_{self.config.col_embedding}': [list(i) for i in score]}
            df_score = pl.DataFrame(dict_)
            df_score.write_parquet(file_name_score)
            df_result = pl.DataFrame(result).drop([self.config.col_embedding])
            df_result.write_parquet(file_name_result)

            # log
            logger.info(
                f"Batch {i}/{num_batches} match result shape: {df_result.shape} "
                f"{perf_counter() - start_batch:,.2f}s"
            )

            # track errors
            if df_result.shape[0] == 0:
                logger.warning(f"Errors")
                return None

            del score, result, df_score, df_result

        # Post process
        df_score = (
            pl.concat([pl.read_parquet(f) for f in sorted(self.config.path_result.glob('score*.parquet'))])
        )
        df_result = (
            pl.concat([pl.read_parquet(f) for f in sorted(self.config.path_result.glob('result*.parquet'))])
        )
        df_match = pl.concat([df_q, df_result, df_score], how='horizontal')
        col_explode = [i for i in df_match.columns if search('db|score', i)]
        df_match = df_match.explode(col_explode)

        return df_match

from pathlib import Path
from pydantic import BaseModel, Field, computed_field
import polars as pl
import duckdb
import re
from time import perf_counter
from core_pro.ultilities import rm_all_folder, make_dir
from .pipeline.build_index_and_query import BuildIndexAndQuery, ConfigQuery
from .pipeline.data_loading import DataEmbedding, ConfigEmbedding
from rich import print


class MatchInput(BaseModel):
    ROOT_PATH: Path = Field(default=None)
    PATH_Q: Path = Field(default=None)
    PATH_DB: Path = Field(default=None)
    PATH_INNER: Path = Field(default=None)
    INNER: bool = Field(default=False)
    MATCH_BY: str = Field(default='text') # image
    COL_CATEGORY: str = Field(default='')
    SHARD_SIZE: int = Field(default=1_500_000)
    QUERY_SIZE: int = Field(default=50_000)
    TOP_K: int = Field(default=10)
    EXPLODE: bool = Field(default=True)

    @computed_field
    @property
    def path_result(self) -> Path:
        path_result = self.ROOT_PATH / f'result_match_{self.MATCH_BY}'
        make_dir(path_result)
        return path_result


class PipelineMatch:
    def __init__(self, record: MatchInput):
        # path
        self.PATH_Q = record.PATH_Q
        self.PATH_DB = record.PATH_DB
        self.PATH_INNER = record.PATH_INNER
        self.ROOT_PATH = record.ROOT_PATH
        self.PATH_RESULT = record.path_result

        # config

        self.COL_CATEGORY = record.COL_CATEGORY
        self.MATCH_BY = record.MATCH_BY
        self.SHARD_SIZE = record.SHARD_SIZE
        self.QUERY_SIZE = record.QUERY_SIZE
        self.TOP_K = record.TOP_K
        self.INNER = record.INNER
        self.EXPLODE = record.EXPLODE

        self.lst_category = self._category_chunking()

    def _category_chunking(self) -> list:
        """
        Read by duckdb to perform lazy load
        """
        query = f"""
        select distinct {{1}}{self.COL_CATEGORY} as category 
        from read_parquet('{{0}}') 
        where {{1}}{self.COL_CATEGORY} is not null
        """
        query = query.format(self.PATH_INNER, '') if self.INNER else query.format(self.PATH_Q, 'q_')
        lst_category = duckdb.sql(query).pl()['category'].to_list()
        return sorted(lst_category)

    def _load_data(self, cat: str, mode: str, file: Path):
        """
        Read by polars to prevent special characters in writing query
        """
        filter_ = pl.col(f'{self.COL_CATEGORY}') == cat if self.INNER else pl.col(f'{mode}_{self.COL_CATEGORY}') == cat
        return pl.read_parquet(file).filter(filter_)

    def run(self):
        # run
        start = perf_counter()
        for idx, cat in enumerate(self.lst_category):
            # logging
            cat_log = f'[dark_orange]{cat}[/]'
            batch_log = f'{idx}/{len(self.lst_category) - 1}'

            # check file exists
            file_result_final = self.PATH_RESULT / f'{cat}.parquet'
            if file_result_final.exists():
                print(f'[PIPELINE] File {cat_log} already exists')
                continue

            # chunk checking
            print('*' * 50)
            print(
                f"🐋 [PIPELINE MATCH BY {self.MATCH_BY}] 🐋 \n"
                f"-> Category: {cat_log} {batch_log} \n"
            )
            if not self.INNER:
                chunk_db = self._load_data(cat=cat, mode='db', file=self.PATH_DB)
                chunk_q = self._load_data(cat=cat, mode='q', file=self.PATH_Q)
                print(f"-> Database shape {chunk_db.shape}, Query shape {chunk_q.shape}")

                if chunk_q.shape[0] < 2 or chunk_db.shape[0] < 2:
                    print(f'[PIPELINE] Database/Query have not enough data')
                    continue

                # embeddings
                config = ConfigEmbedding(
                    ROOT_PATH=self.ROOT_PATH,
                    MODE='db',
                    SHARD_SIZE=self.SHARD_SIZE,
                    MATCH_BY=self.MATCH_BY
                )
                DataEmbedding(config_input=config).load(data=chunk_db)

                config = ConfigEmbedding(
                    ROOT_PATH=self.ROOT_PATH,
                    MODE='q',
                    SHARD_SIZE=self.SHARD_SIZE,
                    MATCH_BY=self.MATCH_BY
                )
                DataEmbedding(config_input=config).load(data=chunk_q)

            else:
                chunk_df = self._load_data(cat=cat, mode='db', file=self.PATH_INNER)
                print(f"-> Inner Data shape {chunk_df.shape}")

                if chunk_df.shape[0] < 2:
                    print(f'[PIPELINE] Database/Query have no data')
                    continue

                # embeddings
                config = ConfigEmbedding(
                    ROOT_PATH=self.ROOT_PATH,
                    MODE='',
                    SHARD_SIZE=self.SHARD_SIZE,
                    MATCH_BY=self.MATCH_BY
                )
                DataEmbedding(config_input=config).load(data=chunk_df)

            # index and query
            config = ConfigQuery(
                ROOT_PATH=self.ROOT_PATH,
                QUERY_SIZE=self.QUERY_SIZE,
                MATCH_BY=self.MATCH_BY,
                TOP_K=self.TOP_K
            )
            cat = re.sub('/', '', cat)  # special characters
            build = BuildIndexAndQuery(
                config=config,
                file_export_name=cat,
                inner=self.INNER,
                explode=self.EXPLODE
            )
            build.build()
            build.query()

            folder_list = ['index', 'result', 'db_array', 'db_ds', 'q_array', 'q_ds', 'array', 'ds']
            for name in folder_list:
                rm_all_folder(self.ROOT_PATH / name)

        time_perf = perf_counter() - start
        print(
            f"🐋 [PIPELINE MATCH BY {self.MATCH_BY}] 🐋 \n"
            f'-> Your files are ready, please find here: {self.PATH_RESULT}'
        )
        return {f'time_perf_{self.MATCH_BY}': time_perf}

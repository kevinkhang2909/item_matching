from pathlib import Path
from pydantic import BaseModel, Field, computed_field
import duckdb
import re
from time import perf_counter
from item_matching.func.utilities import make_dir, rm_all_folder
from src.item_matching.pipeline.build_index_and_query_v2 import BuildIndexAndQuery, ConfigQuery
from src.item_matching.pipeline.data_loading_v2 import DataEmbedding, ConfigEmbedding
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

        # config
        self.COL_CATEGORY = record.COL_CATEGORY
        self.MATCH_BY = record.MATCH_BY
        self.SHARD_SIZE = record.SHARD_SIZE
        self.QUERY_SIZE = record.QUERY_SIZE
        self.TOP_K = record.TOP_K
        self.INNER = record.INNER
        self.EXPLODE = record.EXPLODE

        self.lst_category = self._category_chunking()

    def _category_chunking(self):
        # read query file to extract category
        query = f"select distinct q_{self.COL_CATEGORY} as category from read_parquet('{self.PATH_Q}')"
        if self.INNER:
            query = f"select distinct {self.COL_CATEGORY} as category from read_parquet('{self.PATH_INNER}')"
        return duckdb.sql(query).pl()['category'].to_list()

    def _load_data(self, cat: str, mode: str, file: Path):
        query = f"select * from read_parquet('{file}') where {mode}_{self.COL_CATEGORY} = '{cat}'"
        if self.INNER:
            query = f"select * from read_parquet('{file}') where {self.COL_CATEGORY} = '{cat}'"
        return duckdb.sql(query).pl()

    def run(self):
        # run
        start = perf_counter()
        path_file_result = None
        for idx, cat in enumerate(self.lst_category):
            # chunk checking
            if not self.INNER:
                chunk_db = self._load_data(cat=cat, mode='db', file=self.PATH_DB)
                chunk_q = self._load_data(cat=cat, mode='q', file=self.PATH_Q)

                print(
                    f"🐋 [MATCH BY {self.MATCH_BY}] 🐋 \n"
                    f"-> Category: [dark_orange]{cat}[/] {idx}/{len(self.lst_category)} \n"
                    f"-> Database shape {chunk_db.shape}, Query shape {chunk_q.shape}"
                )

                if chunk_q.shape[0] == 0 or chunk_db.shape[0] == 0:
                    print(f'Database/Query have no data')
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
                DataEmbedding(config_input=config).load(data=chunk_db)

            else:
                chunk_df = self._load_data(cat=cat, mode='db', file=self.PATH_INNER)

                print(
                    f"🐋 [MATCH BY {self.MATCH_BY}] 🐋 \n"
                    f"-> Category: [dark_orange]{cat}[/] {idx}/{len(self.lst_category)} \n"
                    f"-> Inner Data shape {chunk_df.shape}"
                )

                if chunk_df.shape[0] == 0:
                    print(f'Database/Query have no data')
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
            path_file_result = build.query()

            for name in ['index', f'result_match_{self.MATCH_BY}', 'db_array', 'db_ds', 'q_array', 'q_ds']:
                rm_all_folder(self.ROOT_PATH / name)

        time_perf = perf_counter() - start
        print(f'🐋 Your files are ready, please find here: {path_file_result}')
        return {'time_perf': time_perf, 'path_result': path_file_result}

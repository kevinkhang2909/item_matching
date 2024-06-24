from pathlib import Path
from pydantic import BaseModel, Field, computed_field
import duckdb
import re
from time import perf_counter
from item_matching.func.utilities import make_dir, rm_all_folder
from item_matching.pipeline.build_index_and_query import BuildIndexAndQuery, ConfigQuery
from item_matching.pipeline.data_loading import DataEmbedding, ConfigEmbedding

import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format='<level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


class ModelInput(BaseModel):
    ROOT_PATH: Path = Field(default=None)
    PATH_Q: Path = Field(default=None)
    PATH_DB: Path = Field(default=None)
    MATCH_BY: str = Field(default='text')
    COL_CATEGORY: str = Field(default='')
    SHARD_SIZE: int = Field(default=1_500_000)
    QUERY_SIZE: int = Field(default=50_000)
    TOP_K: int = Field(default=10)
    MODE: str = Field(default='')

    @computed_field
    @property
    def path_result(self) -> Path:
        path_result = self.ROOT_PATH / f'result_match'
        make_dir(path_result)
        return path_result


class PipelineMatch:
    def __init__(self, record: ModelInput):
        self.record = record
        self.lst_category = []

    def category_chunking(self):
        # read query file to extract category
        query = f"""
        select distinct q_{self.record.COL_CATEGORY} as category 
        from read_parquet('{self.record.PATH_Q}')
        """
        self.lst_category = duckdb.sql(query).pl()['category'].to_list()

    def load_data(self, cat: str, mode: str = ''):
        file = self.record.model_dump().get(f'PATH_{mode.upper()}')
        query = f"""
        select * 
        from read_parquet('{file}') 
        where {mode}_{self.record.COL_CATEGORY} = '{cat}'
        """
        return duckdb.sql(query).pl()

    def run(self, export_type: str = 'parquet'):
        # extract category
        self.category_chunking()

        # run
        start = perf_counter()
        for idx, cat in enumerate(self.lst_category):
            # chunk checking
            chunk_db = self.load_data(cat, 'db')
            chunk_q = self.load_data(cat, 'q')

            print(
                f"🐋 Start matching by [{self.record.MATCH_BY}] cat: {cat} {idx}/{len(self.lst_category)} - "
                f"Database shape {chunk_db.shape}, Query shape {chunk_q.shape}"
            )

            if chunk_q.shape[0] == 0 or chunk_db.shape[0] == 0:
                print(f'Database/Query have no data')
                continue

            cat = re.sub('/', '', cat)
            file_name = self.record.path_result / f'{cat}.{export_type}'
            if file_name.exists():
                print(f'File already exists: {file_name}')
                continue

            # embeddings
            input_data = self.record.model_copy(update={'MODE': 'db'}).model_dump()
            DataEmbedding(config_input=ConfigEmbedding(**input_data)).load(data=chunk_db)

            input_data = self.record.model_copy(update={'MODE': 'q'}).model_dump()
            DataEmbedding(config_input=ConfigEmbedding(**input_data)).load(data=chunk_q)

            # index and query
            build = BuildIndexAndQuery(config=ConfigQuery(**self.record.model_dump()))
            build.build()
            df_match = build.query(chunk_q)

            # export
            if export_type == 'parquet':
                df_match.write_parquet(file_name)
            else:
                df_match.write_csv(file_name)

            for name in ['index', 'result', 'db_array', 'db_ds', 'q_array', 'q_ds']:
                rm_all_folder(self.record.ROOT_PATH / name)

        time_perf = perf_counter() - start
        print(f'🐋 Your files are ready, please find here: {self.record.path_result}')
        return {'time_perf': time_perf, 'path_result': self.record.path_result}

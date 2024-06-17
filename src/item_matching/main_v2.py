from pathlib import Path
from pydantic import BaseModel, Field, computed_field
import duckdb
from src.item_matching.build_index.build_index_and_query import BuildIndexAndQuery
from src.item_matching.build_index.data_loading import DataEmbedding


class RecordInput(BaseModel):
    ROOT_PATH: Path = Field(default=None)
    PROCESS_PATH_Q: Path = Field(default=None)
    PROCESS_PATH_DB: Path = Field(default=None)
    MATCH_BY: str = Field(default='text')


class PipelineMatch:
    def __init__(self, record: RecordInput):
        self.record = record
        self.lst_category = []

    def category_chunking(self):
        # read query file to extract category
        query = f"""
        select distinct q_{self.record.COL_CATEGORY} as category 
        from read_parquet('{self.record.PATH_Q}')
        """
        self.lst_category = duckdb.sql(query).pl()['category'].to_list()


record = {
    'download_images': False,
    'match_by': 'text_dense',
    'process_path_db': Path('/media/kevin/75b198db-809a-4bd2-a97c-e52daa6b3a2d/item_match/db_clean.parquet'),
    'process_path_q': Path('/media/kevin/75b198db-809a-4bd2-a97c-e52daa6b3a2d/item_match/q_clean.parquet')
}


# data
path = Path('/media/kevin/75b198db-809a-4bd2-a97c-e52daa6b3a2d/item_match')
path_database = path / 'db_clean.parquet'
query = f"""
select *
from read_parquet('{path_database}')
limit 10_000
"""
data = duckdb.sql(query).pl()

# config
input_data_embedding = {
    'SHARD_SIZE': 1_500_000,
    'BATCH_SIZE': 50_000,
    'LEN_DATA': len(data),
    'ROOT_PATH': path,
    'MODE': 'db'
}
config_input = DataInput(**input_data_embedding)

# embedding
DataEmbedding(config_input=config_input).load(data=data)

path_ds = Path('/media/kevin/75b198db-809a-4bd2-a97c-e52daa6b3a2d/item_match/db_ds')
dataset_db = concatenate_datasets([
    load_from_disk(str(f)) for f in sorted(path_ds.glob('*'))
])

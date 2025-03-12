import duckdb
from core_pro.ultilities import make_sync_folder
from pathlib import Path
from datasets import Dataset
import numpy as np
import sys
sys.path.extend([str(Path.home() / 'PycharmProjects/item_matching')])

from src.item_matching.model.model import Model


# path
cluster = 'FMCG'
path = make_sync_folder("dataset/item_matching")

file = path / f'data_sample_{cluster}_clean.parquet'
file_inner = path / 'inner.parquet'

# data loading
query = f"""
with base as (
select * exclude(level1_global_be_category, level2_global_be_category, level3_global_be_category)
, level1_global_be_category || '__' || level2_global_be_category || '__' || level3_global_be_category category
from read_parquet('{file}')
)
select *
from base
"""
df = (
    duckdb.sql(query).pl()
    .unique(['item_id'])
)
dataset_chunk = Dataset.from_polars(df)

model = Model()
model.get_text_model()
embeddings = model.process_text(dataset_chunk["item_name_clean"])
np.save(path / "embeddings.npy", embeddings)

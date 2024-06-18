from src.item_matching import Matching
from pathlib import Path
import polars as pl
from src.item_matching.func.function_image import PipelineImage


path = Path.home() / 'Downloads/item_match'

# db
path_db = path / 'db_0.parquet'
pipe = PipelineImage(path)
df_db = pl.read_parquet(path_db)
df_db, df_img_db = pipe.run(df_db, mode='db', download=False)
df_db.write_parquet(path / 'db_image_clean.parquet')

# q
df_q = df_db.clone()
df_q.columns = [f'q_{i.split('db_')[1]}' for i in df_db.columns]
df_q.write_parquet(path / 'q_image_clean.parquet')

json_stats = Matching(
    col_category='level1_kpi_category',
    path=path,
    path_query=path / 'q_image_clean.parquet',
    path_database=path / 'db_image_clean.parquet',
    query_batch_size=50_000,
    match_mode='image',
).run(export_type='csv')

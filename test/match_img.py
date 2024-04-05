from src.item_matching import Matching
from pathlib import Path
import polars as pl
from src.item_matching.build_index.func_img import PipelineImage


path = Path.home() / 'Downloads/item_match'

# db
path_db = path / 'db_0.parquet'
pipe = PipelineImage(path)
df_db = pl.read_parquet(path_db)
df_db, df_img_db = pipe.run(df_db, mode='db', download=False)

# q
df_q = df_db.clone()
df_q.columns = [f'q_{i.split('db_')[1]}' for i in df_db.columns]

json_stats = Matching(
    col_category='level1_kpi_category',
    path=path,
    df_q=df_q,
    df_db=df_db,
    query_batch_size=300_000
).run(match_mode='image', export_type='csv')

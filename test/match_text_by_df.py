from src.item_matching import Matching
from pathlib import Path
import polars as pl


path = Path.home() / 'Downloads/item_match'
path_db = path / 'db_0.parquet'
path_q = path / 'db_0.parquet'
db = pl.read_csv(path_db)
q = pl.read_csv(path_q)
json_stats = Matching(
    col_category='level1_kpi_category',
    path=path,
    df_db=db,
    df_q=q
).run(match_mode='text')

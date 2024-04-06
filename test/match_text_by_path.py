from src.item_matching import Matching
from pathlib import Path


path = Path.home() / 'Downloads/item_match'
path_db = path / 'db_0.parquet'
path_q = path / 'db_0.parquet'
json_stats = Matching(
    col_category='level1_kpi_category',
    path=path,
    path_database=path_db,
    path_query=path_q,
    match_mode='text',
).run()

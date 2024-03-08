from src.item_matching import Matching
from pathlib import Path
import polars as pl


path = Path('/home/kevin/Downloads/yang')
path_q = path / 'fss (query)/fss_itemid_Home & Living.csv'
path_db = path / 'nonfss (database)/nonfss_itemid_Home & Living.csv'
db = pl.read_csv(path_db)
q = pl.read_csv(path_q)
json_stats = Matching(path, df_q=q, df_db=db).run(match_mode='text')

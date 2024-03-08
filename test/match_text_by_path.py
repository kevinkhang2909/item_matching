from src.item_matching import Matching
from pathlib import Path


path = Path('/home/kevin/Downloads/yang')
path_db = path / 'fss (query)/fss_itemid_Beauty.csv'
path_q = path / 'nonfss (database)/nonfss_itemid_Beauty.csv'
json_stats = Matching(path, path_db, path_q).run(match_mode='text')

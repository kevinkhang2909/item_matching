from src.item_matching import Matching
from pathlib import Path


path = Path('/home/kevin/Downloads/yang')
path_db = '/home/kevin/Downloads/yang/fss (query)/fss_itemid_Beauty.csv'
path_q = '/home/kevin/Downloads/yang/nonfss (database)/nonfss_itemid_Beauty.csv'
Matching(path, path_db, path_q)

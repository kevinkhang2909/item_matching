from tqdm import tqdm
from pathlib import Path
import json
import orjson
import gc

path = Path.home() / '/media/kevin/75b198db-809a-4bd2-a97c-e52daa6b3a2d/item_match/download_img/img_db'
lst_json = sorted(path.glob('*/*.json'))
gc.collect()
[json.loads(open(str(i), "r").read())['url'] for i in tqdm(lst_json, desc='Loading json in folder')]
gc.collect()
[orjson.loads(open(str(i), "r").read())['url'] for i in tqdm(lst_json, desc='Loading json in folder')]

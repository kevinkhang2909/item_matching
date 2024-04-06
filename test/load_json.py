from tqdm import tqdm
from pathlib import Path
import json
import orjson


path = Path.home() / 'Downloads/item_match/download_img/img_db'
lst_json = sorted(path.glob('*/*.json'))
[json.loads(open(str(i), "r").read())['url'] for i in tqdm(lst_json, desc='Loading json in folder')]
[orjson.loads(open(str(i), "r").read())['url'] for i in tqdm(lst_json, desc='Loading json in folder')]

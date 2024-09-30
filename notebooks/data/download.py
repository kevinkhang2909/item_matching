from core_pro import DataPipeLine
from pathlib import Path


path = Path.home() / 'Downloads/Data/Item_Matching_Test'
if str(Path.home()) != '/Users/kevinkhang':
    path = Path('/media/kevin/data_4t/Test')

file = path / 'Description_Images.parquet'
sql = open('./data.sql').read()
df = DataPipeLine(sql).run_presto_to_df(save_path=file)

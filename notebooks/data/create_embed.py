import duckdb
from core_pro.ultilities import make_sync_folder
from src.item_matching.func.function_image import PipelineImage


# path
path = make_sync_folder('Item_Matching_Test')
file = path / 'Description_Images_0.parquet'

query = f"""select * from read_parquet('{file}')"""
df = duckdb.sql(query).pl()

df, df_img = PipelineImage(path).run(df)
df = df.unique(['item_id'])
df.write_parquet(path / 'clean.parquet')

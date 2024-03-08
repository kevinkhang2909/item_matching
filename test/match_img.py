from src.item_matching import Matching
from src.item_matching.build_index.func import load_images
from pathlib import Path
import polars as pl
import duckdb


path = Path('/home/kevin/Downloads/cb')
path_db = path / 'cb_2024-03-07.parquet'

query = f"""
select *
,concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) image_url
from parquet_scan('{str(path_db)}')
order by item_id, images
"""
df_db = (
    duckdb.sql(query).pl()
    # .head(10_000)
)
df_img_db = load_images(path / 'img_cb_2024-03-07', 'db', 'image_url')
df_db = (
    df_db.drop(['images'])
    .select(pl.all().name.prefix(f'db_'))
    .join(df_img_db, on='db_image_url', how='left')
    .filter(pl.col('db_exists'))
)
print(df_db.shape)

df_q = (
    duckdb.sql(query).pl()
    # .head(10_000)
)
df_img_q = load_images(path / 'img_cb_2024-03-07', 'q', 'image_url')
df_q = (
    df_q.drop(['images'])
    .select(pl.all().name.prefix(f'q_'))
    .join(df_img_q, on='q_image_url', how='left')
    .filter(pl.col('q_exists'))
)
print(df_q.shape)

json_stats = Matching(path, df_q=df_q, df_db=df_db).run(match_mode='image', clean_text=False, export_type='csv')

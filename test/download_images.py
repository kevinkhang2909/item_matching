from pathlib import Path
import duckdb
import polars as pl
from src.item_matching.build_index.func import download_images


# path
path = Path.home() / 'Downloads/cb'

# data
name = 'cb_2024-03-07.parquet'
query = f"""
select *
,concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) image_url
from parquet_scan('{str(path / name)}')
order by item_id, images
"""
df = (
    duckdb.sql(query).pl()
    .with_columns(pl.col('create_datetime').dt.date().cast(pl.Utf8))
)
print(df.shape, df['item_name'].n_unique())


# download
path_parq = path / f'{name}_0.parquet'
df.write_parquet(path_parq)
download_images(path, name, 'image_url')

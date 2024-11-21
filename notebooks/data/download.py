import os, subprocess
from rich import print
import duckdb
import polars as pl
from core_pro import DataPipeLine
from core_pro.ultilities import make_sync_folder, make_dir
from core_eda import EDA_Dataframe


# download parquet
path = make_sync_folder('Item_Matching_Test')

file = path / 'Description_Images.parquet'
if not file.exists():
    print(f'File {file.stem} does not exist')
    sql = open('./data.sql').read()
    df = DataPipeLine(sql).run_presto_to_df(save_path=file)
else:
    df = pl.read_parquet(file)
    print(f'File {file.stem} exists')
print(df.head(10))

# download images
name = 'Description_Images'
query = f"""
select *
,concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) image_url
from parquet_scan('{str(path / f'{name}.parquet')}')
order by item_id, images
"""
df = (
    duckdb.sql(query).pl()
    .unique(['item_id'])
)
EDA_Dataframe(df, ['item_id']).check_duplicate()

path_parq = path / f'{name}_0.parquet'
df.write_parquet(path_parq)

path_image = path / 'download_img'
make_dir(path_image)
folder = path_image / f'img'
if not folder.exists():
    os.chdir(str(path_image))
    command = (
        f"img2dataset --url_list={str(path_parq)} "
        f"--output_folder=img_/ "
        f"--processes_count=16 "
        f"--thread_count=32 "
        f"--image_size=224 "
        f"--output_format=files "
        f"--input_format=parquet "
        f"--url_col=image_url "
        f"--number_sample_per_shard=50000 "
    )
    subprocess.run(command, shell=True)

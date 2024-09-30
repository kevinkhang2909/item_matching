from pathlib import Path
import duckdb
import os
import subprocess
from core_pro.ultilities import make_dir


# path
path = Path.home() / 'Downloads/Data/Item_Matching_Test'
if str(Path.home()) != '/Users/kevinkhang':
    path = Path('/media/kevin/data_4t/Test')

# data
name = 'Description_Images'
query = f"""
select *
,concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) image_url
from parquet_scan('{str(path / f'{name}.parquet')}')
order by item_id, images
"""
df = duckdb.sql(query).pl()
print(df.shape, df['item_name'].n_unique())

# download
path_parq = path / f'{name}_0.parquet'
df.write_parquet(path_parq)

path_image = path / 'download_img'
make_dir(path_image)
folder = path_image / f'img'
if not folder.exists():
    os.chdir(str(path_image))
    command = (
        f"img2dataset --url_list={str(path_parq)} "
        f"--output_folder=img_{name}/ "
        f"--processes_count=16 "
        f"--thread_count=32 "
        f"--image_size=224 "
        f"--output_format=files "
        f"--input_format=parquet "
        f"--url_col=image_url "
        f"--number_sample_per_shard=50000 "
    )
    subprocess.run(command, shell=True)

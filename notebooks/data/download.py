from rich import print
import duckdb
import polars as pl
from core_pro import DataPipeLine
from core_pro.ultilities import make_sync_folder
from core_eda import EDA_Dataframe
from pathlib import Path
import sys

sys.path.extend([str(Path.home() / "PycharmProjects/item_matching")])

from src.item_matching import PipelineImage


# download parquet
path = make_sync_folder("dataset/item_matching")
path_sql = Path.home() / "PycharmProjects/item_matching/notebooks/data/data.sql"
path_image = path / "img"

# for c in ['ELHA', 'Lifestyle', 'Fashion', 'FMCG']:
for c in ["FMCG"]:
    file = path / f"data_sample_{c}.parquet"
    if not file.exists():
        print(f"Download {file.stem}")
        sql = open(str(path_sql)).read().format(c)
        df = DataPipeLine(sql).run_presto_to_df(save_path=file)
    else:
        df = pl.read_parquet(file)
        print(f"File {file.stem} exists")

    # download images
    query = f"""
    select *
    ,concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) image_url
    from read_parquet('{str(path / f'{file.stem}.parquet')}')
    order by item_id, images
    """
    df = duckdb.sql(query).pl().unique(["item_id"])
    EDA_Dataframe(df, ["item_id"]).check_duplicate()

    df, df_img = PipelineImage(
        path_image=path_image, mode="", col_img_download="image_url"
    ).run(df, download=True, num_workers=4, num_processes=4)
    df.write_parquet(path / f"data_sample_{c}_clean.parquet")

from pathlib import Path
import pandas as pd
import polars as pl
from core_pro.ultilities import make_dir
import sys
from loguru import logger
from tqdm import tqdm
from item_matching.build_index.func import clean_text

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


class PipelineImage:
    def __init__(
            self,
            path: Path,
            col_image: str = 'image_url',
    ):
        self.path_image = path / 'download_img'
        self.col_image = col_image

        # init path image
        make_dir(self.path_image)

    def download_images(self, file_name: str):
        import os
        import subprocess

        os.chdir(str(self.path_image))
        command = (
            f"img2dataset --url_list={file_name}_0.parquet "
            f"--output_folder=img_{file_name}/ "
            f"--processes_count=16 "
            f"--thread_count=32 "
            f"--image_size=224 "
            f"--output_format=files "
            f"--input_format=parquet "
            f"--url_col={self.col_image} "
            f"--number_sample_per_shard=50000 "
        )
        subprocess.run(command, shell=True)

    def load_images(self, mode: str = '') -> pl.DataFrame:
        import orjson

        # listing
        path = self.path_image / f'img_{mode}'
        lst_json = sorted(path.glob('*/*.json'))
        lst_file = [orjson.loads(open(str(i), "r").read())['url'] for i in tqdm(lst_json, desc='Loading json in folder')]
        lst_img = [str(i) for i in tqdm(sorted(path.glob('*/*.jpg')), desc='Loading jpg in folder')]
        df = pl.DataFrame({
            f'{mode}_{self.col_image}': lst_file,
            f'{mode}_file_path': lst_img,
            f'{mode}_exists': [True] * len(lst_file),
        })

        logger.info(f'[Data] Load Images: {df.shape}')
        return df

    def run(
            self,
            data: pl.DataFrame | pd.DataFrame,
            mode: str = '',
            download: bool = False,
    ):
        import duckdb

        # load data
        query = f"""
        select *
        ,concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) {self.col_image}
        from data
        """
        df = duckdb.sql(query).pl()
        logger.info(f'[Data] Base Data: {df.shape}')

        # download
        if download:
            df.write_parquet(self.path_image / f'{mode}_0.parquet')
            self.download_images(mode)

        # load data image
        data_img = self.load_images(mode)

        # join
        data = (
            df.drop(['images'])
            .pipe(clean_text)
            .select(pl.all().name.prefix(f'{mode}_'))
            .join(data_img, on=f'{mode}_{self.col_image}', how='left')
            .filter(pl.col(f'{mode}_exists'))
        )
        logger.info(f'[Data] Join Images: {data.shape}')
        return data, data_img

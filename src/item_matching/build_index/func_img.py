from pathlib import Path
import polars as pl
import duckdb
import orjson
from tqdm import tqdm
import subprocess
import os
import sys
from loguru import logger
from .func import PipelineText, make_dir

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


class PipelineImage:
    def __init__(
            self,
            path: Path,
            col_image: str = 'image_url',
            mode: str = ''
    ):
        self.path_image = path / 'download_img'
        self.col_image = col_image
        self.mode = mode

        # init path image
        make_dir(self.path_image)

    def download_images(self):
        folder = self.path_image / f'img_{self.mode}'
        if not folder.exists():
            os.chdir(str(self.path_image))
            command = (
                f"img2dataset --url_list={self.mode}_0.parquet "
                f"--output_folder=img_{self.mode}/ "
                f"--processes_count=16 "
                f"--thread_count=32 "
                f"--image_size=224 "
                f"--output_format=files "
                f"--input_format=parquet "
                f"--url_col={self.col_image} "
                f"--number_sample_per_shard=50000 "
            )
            subprocess.run(command, shell=True)

    def load_images(self) -> pl.DataFrame:
        # listing
        path = self.path_image / f'img_{self.mode}'
        lst_json = sorted(path.glob('*/*.json'))
        lst_file = [orjson.loads(open(str(i), "r").read())['url'] for i in tqdm(lst_json, desc='Loading json in folder')]
        lst_img = [str(i) for i in tqdm(sorted(path.glob('*/*.jpg')), desc='Loading jpg in folder')]
        df = pl.DataFrame({
            f'{self.mode}_{self.col_image}': lst_file,
            f'{self.mode}_file_path': lst_img,
            f'{self.mode}_exists': [True] * len(lst_file),
        })

        logger.info(f'[Data] Load Images: {df.shape}')
        return df

    def run(
            self,
            data,
            download: bool = False,
            edit_img_url: bool = True,
    ):
        # edit url
        col_query = f",concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) {self.col_image}"
        if edit_img_url:
            col_query = f"images {self.col_image}"

        # load data
        query = f"""select *, {col_query} from data"""
        df = duckdb.sql(query).pl()
        logger.info(f'[Data] Base Data {self.mode}: {df.shape}')

        # download
        if download:
            df.write_parquet(self.path_image / f'{self.mode}_0.parquet')
            self.download_images()

        # load data image
        data_img = self.load_images()

        # join
        data = (
            df.drop(['images'])
            .pipe(PipelineText.clean_text)
            .select(pl.all().name.prefix(f'{self.mode}_'))
            .join(data_img, on=f'{self.mode}_{self.col_image}', how='left')
            .filter(pl.col(f'{self.mode}_exists'))
        )
        logger.info(f'[Data] Join Images {self.mode}: {data.shape}')
        return data, data_img

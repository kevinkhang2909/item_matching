from pathlib import Path
import polars as pl
import duckdb
import orjson
from tqdm import tqdm
import subprocess
import os
import sys
import requests
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFile, UnidentifiedImageError
from .function_text import PipelineText
from item_matching.func.utilities import make_dir

logger.remove()
logger.add(sys.stdout, colorize=True, format='<level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')
ImageFile.LOAD_TRUNCATED_IMAGES = True


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

    def download_images_request(self):
        def _download(arr: dict):
            # init
            url = arr['url']
            idx = arr['index']

            # get
            response = requests.get(url, stream=True)

            # path
            path_img = folder / f'{idx}.jpg'
            path_json = folder / f'{idx}.json'

            # resize and log json
            if not path_img.exists():
                try:
                    # img
                    img = Image.open(response.raw).convert('RGB').resize((224, 224))
                    img.save(str(path_img))
                    # json
                    json_object = orjson.dumps(arr, option=orjson.OPT_INDENT_2).decode("utf-8")
                    with open(str(path_json), 'w') as outfile:
                        outfile.write(json_object)
                except UnidentifiedImageError as e:
                    pass

        # path
        folder = self.path_image / f'img_{self.mode}/00000'
        if not folder.exists():
            make_dir(folder)
            # read data
            query = f"""select * from read_parquet('{self.path_image}/{self.mode}_0.parquet')"""
            df = (
                duckdb.sql(query).pl()
                .rename({self.col_image: 'url'})
                .with_row_index()
            )
            # run
            run = df.to_dicts()
            with ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(_download, run), total=len(run)))

    def download_images_img2dataset(self):
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
            download_mode: str = 'img2dataset',
            edit_img_url: bool = True,
    ):
        # edit url
        col_query = f",concat('http://f.shopee.vn/file/', UNNEST(array_slice(string_split(images, ','), 1, 1))) {self.col_image}"
        if edit_img_url:
            col_query = f"images {self.col_image}"

        # load data
        query = f"""select *, {col_query} from data"""
        df = duckdb.sql(query).pl()
        df.write_parquet(self.path_image / f'{self.mode}_0.parquet')
        logger.info(f'[Data] Base Data {self.mode}: {df.shape}')

        # download
        if download:
            self.download_images_img2dataset() if download_mode == 'img2dataset' else self.download_images_request()

        # load data image
        data_img = self.load_images()

        # errors image
        if data_img.shape[0] == 0:
            logger.info(f'[Data] Images Errors {self.mode}: {data.shape}')
            return data, data_img

        # join
        data = (
            df.drop(['images'])
            .pipe(PipelineText.clean_text)
            .join(data_img, on=f'{self.mode}_{self.col_image}', how='left')
            .filter(pl.col(f'{self.mode}_exists'))
        )
        logger.info(f'[Data] Join Images {self.mode}: {data.shape}')
        return data, data_img

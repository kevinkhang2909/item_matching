from pathlib import Path
import polars as pl
import duckdb
import orjson
from tqdm.auto import tqdm
import subprocess
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFile, UnidentifiedImageError
from rich import print
from core_pro.ultilities import make_dir
from .function_text import PipelineText

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PipelineImage:
    def __init__(
            self,
            path_image: Path,
            mode: str = '',
            col_img_download: str = 'image_url',
    ):
        self.path_image = path_image
        self.col_img_download = col_img_download
        self.mode = mode

        # init path image
        make_dir(self.path_image)
        print(f'[Image Cleaning] {mode}')

    def download_images_request(self):
        def _download(arr: dict):
            # init
            url = arr[self.col_img_download]
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
                f"--url_col={self.col_img_download} "
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
            f'{self.col_img_download}': lst_file,
            f'file_path': lst_img,
            f'exists': [True] * len(lst_file),
        })

        print(f'-> Load Images Data from folder: {df.shape}')
        return df

    def run(
            self,
            data,
            download: bool = False,
            download_mode: str = 'img2dataset',
    ):
        # load data
        query = f"""select * from data"""
        df = duckdb.sql(query).pl()
        df.write_parquet(self.path_image / f'{self.mode}_0.parquet')
        print(f'-> Base Data {self.mode}: {df.shape}')

        # download
        if download:
            self.download_images_img2dataset() if download_mode == 'img2dataset' else self.download_images_request()

        # load data image
        data_img = self.load_images()

        # errors image
        if data_img.shape[0] == 0:
            print(f'-> Images Errors {self.mode}: {data.shape}')
            return data, data_img

        # join
        data = (
            df
            .pipe(PipelineText().clean_text)
            .join(data_img, on=f'{self.col_img_download}', how='left')
            .filter(pl.col(f'exists'))
        )
        if self.mode != '':
            data = data.select(pl.all().name.prefix(f'{self.mode}_'))

        print(f'-> Clean Images Data {self.mode}: {data.shape}')
        return data, data_img

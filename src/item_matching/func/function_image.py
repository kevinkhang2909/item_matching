from pathlib import Path
import polars as pl
from rich import print
from core_pro import ImageDownloader
from core_pro.ultilities import make_dir
from .function_text import PipelineText


class PipelineImage:
    def __init__(
            self,
            path_image: Path,
            mode: str = '',
            col_img_download: str = 'image_url',
            col_text: str = 'item_name'
    ):
        # path
        self.path_image = path_image
        self.mode = mode
        self.folder_image = self.path_image / f'img_{self.mode}'

        # config
        self.col_img_download = col_img_download
        self.col_text = col_text
        make_dir(self.path_image)

        print(f'[Image Processing] {mode}')

    def load_images(self) -> pl.DataFrame:
        lst = [(str(i), int(i.stem)) for i in self.folder_image.glob('*.jpg')]
        df = (
            pl.DataFrame(lst, orient='row', schema=['file_path', 'index'])
            .with_columns(pl.col('index').cast(pl.UInt32))
        )
        print(f'-> Load images from folder: {df.shape}')
        return df

    def run(
            self,
            data,
            download: bool = False,
            num_processes: int = 4,
            num_workers: int = 16,
    ):
        # load data
        data = data.with_row_index()
        run = data[['index', self.col_img_download]].to_numpy().tolist()

        print(f'-> Base data {self.mode}: {data.shape}')

        # download
        if download:
            downloader = ImageDownloader(
                output_dir=self.folder_image,
                num_processes=num_processes,
                threads_per_process=num_workers,
                resize_size=(224, 224),
            )
            downloader.download_images(run)

        # load data image
        data_img = self.load_images()

        # errors image
        if data_img.shape[0] == 0:
            print(f'-> Images Errors {self.mode}: {data.shape}')
            return data, data_img

        # join
        data = (
            data
            .pipe(PipelineText().clean_text, col=self.col_text)
            .join(data_img, on='index', how='left')
            .filter(~pl.col('file_path').is_null())
        )
        if self.mode != '':
            data = data.select(pl.all().name.prefix(f'{self.mode}_'))

        print(f'-> Merge and clean images data {self.mode}: {data.shape}')
        return data, data_img

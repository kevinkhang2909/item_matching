from pathlib import Path
import polars as pl
from rich import print
from core_pro import ImageDownloader
from core_pro.ultilities import make_dir
from core_eda import TextEDA


class PipelineImage:
    def __init__(
        self,
        path_image: Path,
        mode: str = "",
        col_img_download: str = "image_url",
        col_text: str = "item_name",
    ):
        # path
        self.path_image = path_image
        self.mode = mode
        self.folder_image = self.path_image / f"img_{self.mode}"

        # config
        self.col_img_download = col_img_download
        self.col_text = col_text
        make_dir(self.path_image)

        print(f"[Image Processing] {mode}")

    def load_images(self) -> pl.DataFrame:
        lst = [(int(i.stem), str(i), i.exists()) for i in self.folder_image.glob("*/*.jpg")]
        df = (
            pl.DataFrame(lst, orient="row", schema=["index", "file_path", "exists"])
            .with_columns(pl.col("index").cast(pl.UInt32))
        )
        if df.shape[0] == 0:
            print(f"-> Images Errors {self.mode}: {df.shape}")
        else:
            print(f"-> Load images from folder {self.mode}: {df.shape}")
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
        run = data[self.col_img_download].to_list()
        print(f"-> Base data {self.mode}: {data.shape}")

        # download
        if download:
            downloader = ImageDownloader(
                output_dir=self.folder_image,
                num_processes=num_processes,
                threads_per_process=num_workers,
                resize_size=(224, 224),
                batch_size=1000,
                images_per_folder=1000,
            )
            downloader.download_images(run)

        # load data image
        data_img = self.load_images()

        # join
        data = (
            data.pipe(TextEDA.clean_text_pipeline_polars, col=self.col_text)
            .join(data_img, on="index", how="left")
            .filter(pl.col("exists"))
        )
        if self.mode != "":
            data = data.select(pl.all().name.prefix(f"{self.mode}_"))

        print(f"-> Merge and clean images data {self.mode}: {data.shape}")
        return data, data_img

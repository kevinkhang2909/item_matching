from pathlib import Path
import polars as pl
from rich import print
from core_pro import ImgDownloaderThreadProcess
from core_pro.ultilities import make_dir
from core_eda import TextEDA


class PipelineImage:
    def __init__(
        self,
        path_image: Path,
        mode: str = "",

    ):
        # path
        self.path_image = path_image
        make_dir(self.path_image)
        self.mode = mode
        self.folder_image = self.path_image / f"img_{self.mode}"

        print(f"[Image Processing] {mode}")

    def load_images(self) -> pl.DataFrame:
        # sorted files
        files_sorted = sorted([*self.folder_image.glob("*/*.jpg")], key=lambda x: int(x.stem.split("_")[0]))

        # file to df
        lst = [(i.stem, str(i), i.exists()) for i in files_sorted]
        df = pl.DataFrame(lst, orient="row", schema=["path_idx", "file_path"])
        if df.shape[0] == 0:
            print(f"-> Images Errors {self.mode}: {df.shape}")
        else:
            print(f"-> Load images from folder {self.mode}: {df.shape}")
        return df

    def run(
        self,
        data,
        col_image_url: str = "image_url",
        col_text: str = "item_name",
        download: bool = False,
        num_processes: int = 4,
        num_workers: int = 16,
    ):
        # load data
        data = data.with_row_index("img_index")
        run = [(i["img_index"], i[col_image_url]) for i in data[["img_index", col_image_url]].to_dicts()]
        print(f"-> Base data {self.mode}: {data.shape}")

        # download
        if download:
            downloader = ImgDownloaderThreadProcess(
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
        img_key = [f"{i[0]}_{i[1].split("/")[-1]}" for i in run]
        data = (
            data.pipe(TextEDA.clean_text_pipeline_polars, col=col_text)
            .with_columns(pl.Series("path_idx", img_key))
            .join(data_img, on=["path_idx"], how="left")
            .filter(pl.col("file_exists"))
        )
        if self.mode != "":
            data = data.select(pl.all().name.prefix(f"{self.mode}_"))

        print(f"-> Merge and clean images data {self.mode}: {data.shape}")
        return data, data_img

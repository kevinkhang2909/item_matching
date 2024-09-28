from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Draw:
    def __init__(self):
        pass

    @staticmethod
    def plot_one_image(file_path: str, fig_size: tuple = (7, 3)):
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.imshow(Image.open(file_path))
        fig.tight_layout()

    @staticmethod
    def plot_multiple_images(file_paths: list, fig_size: tuple = (20, 10)):
        # layout
        num_images, ncol = min(len(file_paths), 5), 5
        nrow = int(np.ceil(num_images / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
        axes = axes.flatten()

        # plot
        for i, image in enumerate(file_paths):
            axes[i].imshow(Image.open(image))
            axes[i].axis('off')

        # remove axes
        for _ in range(num_images, ncol * nrow, 1):
            axes[_].remove()
        fig.tight_layout()


def rm_all_folder(path: Path) -> None:
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_all_folder(child)
    path.rmdir()


def make_dir(folder_name: str | Path) -> None:
    """Make a directory if it doesn't exist"""
    if isinstance(folder_name, str):
        folder_name = Path(folder_name)
    if not folder_name.exists():
        folder_name.mkdir(parents=True, exist_ok=True)


def clean_check(data: pl.DataFrame, pct: bool = False, verbose: bool = False) -> dict:
    null_check_dict = data.null_count().to_dicts()[0].items()
    null_check = {i: v for i, v in null_check_dict if v > 0}
    if pct:
        null_check = {i: v / data.shape[0] for i, v in null_check_dict if v > 0}

    if verbose:
        print(f"- Null columns: {null_check}")
    dict_ = {
        'null': null_check,
    }
    return dict_

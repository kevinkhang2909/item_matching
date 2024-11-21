import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import textwrap
import polars as pl
import duckdb
import sys
import re
import emoji
from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format='<level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


class Draw:
    def __init__(self):
        pass

    @staticmethod
    def plot_one_image(file_path: Path, title: str = None, fig_size: tuple = (7, 3)):
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.imshow(Image.open(file_path))
        ax.axis('off')
        if title:
            ax.set_title(textwrap.fill(title, width=30, break_long_words=True), wrap=True, fontsize=10, ha='center')
        fig.tight_layout()
        fig.show()

    @staticmethod
    def plot_one_image_long_caption(file_path: Path, caption: str, fig_size: tuple = (12, 5), vertical: bool = False):
        nrows, ncols = 1, 2
        if vertical:
            nrows, ncols = 2, 1
            fig_size = (5, 12)

        fig, (ax1, ax2) = plt.subplots(nrows, ncols, figsize=fig_size)
        ax1.imshow(Image.open(file_path))
        ax1.axis('off')
        ax2.text(0, 0.5, caption, va='center', ha='left', wrap=True)
        ax2.axis('off')
        fig.tight_layout()
        fig.show()

    @staticmethod
    def plot_multiple_images(file_paths: list, title: list = None, fig_size: tuple = (20, 10)):
        # layout
        num_images, ncol = len(file_paths), min(len(file_paths), 3)
        nrow = int(np.ceil(num_images / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
        axes = axes.flatten()

        # plot
        for i, (image, t) in enumerate(zip(file_paths, title)):
            axes[i].imshow(Image.open(image))
            axes[i].axis('off')
            if title:
                axes[i].set_title(textwrap.fill(t, width=40, break_long_words=True), wrap=True, fontsize=10, ha='center')
        # remove axes
        for _ in range(num_images, ncol * nrow, 1):
            axes[_].remove()
        fig.tight_layout()


class PipelineText:
    def __init__(self, mode: str = ''):
        self.mode = mode

    @staticmethod
    def remove_text_between_emojis(text):
        # regex pattern to match emojis
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        # find all emojis in the text
        emojis = emoji_pattern.findall(text)
        # if there are less than 2 emojis, return the original text
        if len(emojis) < 2:
            return text
        else:
            regex = f"[{emojis[0]}].*?[{emojis[1]}]"
            return re.sub(regex, "", text)

    @staticmethod
    def clean_text_pipeline(text: str) -> str:
        regex = r"[\(\[\<\"\|].*?[\)\]\>\"\|]"
        text = str(text).lower().strip()
        text = PipelineText.remove_text_between_emojis(text)
        text = emoji.replace_emoji(text, ' ')
        text = re.sub(regex, ' ', text)
        text = re.sub(r'\-|\_|\*', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.rstrip('.').strip()

    @staticmethod
    def clean_text(data: pl.DataFrame, col: str = 'item_name') -> pl.DataFrame:
        lst = [PipelineText.clean_text_pipeline(str(x)) for x in tqdm(data[col].to_list(), desc='[Pipeline] Clean Text')]
        return data.with_columns(pl.Series(name=f'{col}_clean', values=lst))

    def run(self, data, key_col: list = None):
        # load data
        query = f"""select * from data"""
        df = duckdb.sql(query).pl()
        logger.info(f'[Data] Base Data {self.mode}: {df.shape}')

        df = (
            df
            .pipe(PipelineText.clean_text)
            .drop_nulls(subset=key_col)
        )
        logger.info(f'[Data] Join Data {self.mode}: {df.shape}')
        return df

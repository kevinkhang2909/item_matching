from pathlib import Path
import polars as pl
import duckdb
import sys
import re
from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


class PipelineText:
    def __init__(self, mode: str = ''):
        self.mode = mode

    @staticmethod
    def clean_text(data: pl.DataFrame, col: str = 'item_name') -> pl.DataFrame:
        regex = "[\(\[\<\"].*?[\)\]\>\"]"
        return data.with_columns(
            pl.col(col).map_elements(
                lambda x: re.sub(regex, "", x).lower().rstrip('.').strip(), return_dtype=pl.String
            )
            .alias(f'{col.lower()}_clean')
        )

    def run(self, data):
        # load data
        query = f"""select * from data"""
        df = duckdb.sql(query).pl()
        logger.info(f'[Data] Base Data {self.mode}: {df.shape}')

        df = (
            df
            .pipe(PipelineText.clean_text)
            .select(pl.all().name.prefix(f'{self.mode}_'))
            .drop_nulls()
        )
        logger.info(f'[Data] Join Images {self.mode}: {df.shape}')
        return df


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

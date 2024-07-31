import polars as pl
import duckdb
import sys
import re
from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=True, format='<level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


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

    def run(self, data, key_col: list = None):
        # load data
        query = f"""select * from data"""
        df = duckdb.sql(query).pl()
        logger.info(f'[Data] Base Data {self.mode}: {df.shape}')

        df = (
            df
            .pipe(PipelineText.clean_text)
            .drop_nulls(subset=key_col)
            .select(pl.all().name.prefix(f'{self.mode}_'))
        )
        logger.info(f'[Data] Join Data {self.mode}: {df.shape}')
        return df

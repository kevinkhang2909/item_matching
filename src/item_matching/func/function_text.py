import polars as pl
import duckdb
from core_eda import TextEDA
from tqdm.auto import tqdm
from rich import print


class PipelineText:
    def __init__(self, mode: str = ''):
        self.mode = mode
        print(f'[Text Cleaning] {mode}')

    @staticmethod
    def clean_text(data: pl.DataFrame, col: str = 'item_name') -> pl.DataFrame:
        lst = [TextEDA.clean_text_pipeline(str(x)) for x in tqdm(data[col].to_list(), desc='Clean Text')]
        return data.with_columns(pl.Series(name=f'{col}_clean', values=lst))

    def run(self, data, key_col: list = None):
        # load data
        query = f"""select * from data"""
        df = duckdb.sql(query).pl()
        print(f'-> Base Data {self.mode}: {df.shape}')

        df = (
            df
            .pipe(PipelineText.clean_text)
            .drop_nulls(subset=key_col)
        )
        if self.mode != '':
            df = df.select(pl.all().name.prefix(f'{self.mode}_'))
        print(f'-> Cleaned Data {self.mode}: {df.shape}')
        return df

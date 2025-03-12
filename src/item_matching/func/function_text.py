import polars as pl
import duckdb
from core_eda import TextEDA
from rich import print


class PipelineText:
    def __init__(self, mode: str = ""):
        self.mode = mode
        print(f"[Text Cleaning] {mode}")

    def run(self, data, key_col: list = None):
        # load data
        query = f"""select * from data"""
        df = duckdb.sql(query).pl()
        print(f"-> Base Data {self.mode}: {df.shape}")

        df = df.pipe(TextEDA.clean_text_pipeline_polars).drop_nulls(subset=key_col)
        if self.mode != "":
            df = df.select(pl.all().name.prefix(f"{self.mode}_"))
        print(f"-> Cleaned Data {self.mode}: {df.shape}")
        return df

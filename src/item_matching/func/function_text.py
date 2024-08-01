import polars as pl
import duckdb
import sys
import re
import emoji
from tqdm import tqdm
from loguru import logger
logger.remove()
logger.add(sys.stdout, colorize=True, format='<level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')


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
        text = text.lower().strip()
        text = PipelineText.remove_text_between_emojis(text)
        text = emoji.replace_emoji(text, ' ')
        text = re.sub(regex, ' ', text)
        text = re.sub(r'\-|\_|\*', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.rstrip('.').strip()

    @staticmethod
    def clean_text(data: pl.DataFrame, col: str = 'item_name') -> pl.DataFrame:
        lst = [PipelineText.clean_text_pipeline(x) for x in tqdm(data[col].to_list())]
        return data.with_columns(pl.Series(name=f'{col}_clean', values=lst))
        # regex = "[\(\[\<\"].*?[\)\]\>\"]"
        # return data.with_columns(
        #     pl.col(col).map_elements(
        #         lambda x: re.sub(regex, "", x).lower().rstrip('.').strip(), return_dtype=pl.String
        #     )
        #     .alias(f'{col.lower()}_clean')
        # )

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

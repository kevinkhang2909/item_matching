import polars as pl
import duckdb
from pathlib import Path
from pydantic import BaseModel, Field, computed_field
from tqdm import tqdm
from rich import print
from core_pro.ultilities import make_dir
from ..func.post_processing import PostProcessing, data_explode_list


class ReRankConfig(BaseModel):
    ROOTPATH: Path = Field(default=None)
    EXPLODE: bool = Field(default=False)

    @computed_field
    @property
    def path_result(self) -> Path:
        path_result = self.ROOTPATH / 'result_match_rerank'
        make_dir(path_result)
        return path_result

    @computed_field
    @property
    def path_text(self) -> Path:
        return self.ROOTPATH / 'result_match_text'

    @computed_field
    @property
    def path_image(self) -> Path:
        return self.ROOTPATH / 'result_match_image'

    @computed_field
    @property
    def file_text(self) -> list:
        return [*self.path_text.glob('*')]

    @computed_field
    @property
    def file_image(self) -> list:
        return [*self.path_image.glob('*')]


class ReRank:
    def __init__(self, record: ReRankConfig):
        # all category
        self.all_category = set([i.stem for i in record.file_text] + [i.stem for i in record.file_image])

        # config
        self.EXPLODE = record.EXPLODE

        # path
        self.path_result = record.path_result
        self.path_image = record.path_image
        self.path_text = record.path_text

        # file
        self.file_text = record.file_text
        self.file_image = record.file_image

    def _data_check(self, category: str):
        paths = {'text': self.path_text, 'image': self.path_image}
        df_dict, total_cols = {}, []
        for name, path in paths.items():
            df = pl.read_parquet(path / f'{category}.parquet')
            if self.EXPLODE:
                df = data_explode_list(df)
            total_cols.extend(df.columns)
            df_dict[name] = df
        total_cols = set(total_cols)

        # select cols
        patterns = ['item_name_clean', 'file_path', 'rnk', 'exists', 'score_text_embed', 'score_image_embed']
        select_cols = [col for col in total_cols if not any(pattern in col for pattern in patterns)]
        duckdb_cols = '\n, '.join([f'{i}' for i in select_cols])
        return df_dict, duckdb_cols

    def rerank_score(self, data_text, data_image, duckdb_cols):
        # query
        query = f"""
            -- combine text & image
            with base as (
                select {duckdb_cols}
                , 'score_text' match_type
                , score_text_embed score
                from data_text
                union all
                select {duckdb_cols}
                , 'score_image' match_type
                , score_image_embed score
                from data_image
            ) 
            -- pivot scores into 2 cols
            , pivot_tab as (
                PIVOT base
                ON match_type
                USING sum(score)
                GROUP BY {duckdb_cols}
            )
            -- calculate mean, max rerank
            , cal_tab as (
                select *
                , (coalesce(score_text, 0) + coalesce(score_image, 0)) / 2 as score_mean
--                 , greatest(score_text, score_image) as score_max
                from pivot_tab
                where db_item_id != q_item_id
            )
            -- rank matches
            select * 
            , row_number() OVER (PARTITION BY q_item_id ORDER BY score_mean desc) ranking
            from cal_tab
            """
        return duckdb.sql(query).pl()

    def run(self):
        total_cat = len(self.all_category)
        for cat in tqdm(self.all_category, total=total_cat, desc='ReRanking'):
            df_dict, duckdb_cols = self._data_check(cat)
            df = self.rerank_score(data_text=df_dict['text'], data_image=df_dict['image'], duckdb_cols=duckdb_cols)
            df = PostProcessing(df).run()
            df.write_parquet(self.path_result / f'{cat}.parquet')
        print(f'[RERANK]: Done {total_cat} categories')

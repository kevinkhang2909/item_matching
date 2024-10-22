import polars as pl
import duckdb
from pathlib import Path
from core_pro.ultilities import make_dir
from pydantic import BaseModel, Field, computed_field
from ..func.post_processing import PostProcessing


class ReRankConfig(BaseModel):
    path_text_result: Path = Field(default=None)
    path_image_result: Path = Field(default=None)

    @computed_field
    @property
    def path_result(self) -> Path:
        path_result = self.path_text_result.parent
        make_dir(path_result)
        return path_result

    @computed_field
    @property
    def file_text(self) -> list:
        return [*self.path_text_result.glob('*')]

    @computed_field
    @property
    def file_image(self) -> list:
        return [*self.path_image_result.glob('*')]


class ReRank:
    def __init__(self, record: ReRankConfig):
        # path
        self.path_img = record.path_text_result
        self.path_text = record.path_text_result
        self.path_result = record.path_result

        # file
        self.file_text = record.file_text
        self.file_image = record.file_image

        # all category
        self.all_category = set([i.stem for i in self.file_text] + [i.stem for i in self.file_image])

        # select cols
        total_cols = set(pl.read_parquet(self.file_text[0]).columns + pl.read_parquet(self.file_image[0]).columns)
        patterns = ['item_name_clean', 'file_path', 'rnk', 'exists']
        select_cols = [col for col in total_cols if not any(pattern in col for pattern in patterns)]
        self.duckdb_cols = '\n, '.join([f'{i}' for i in select_cols])

    def rerank_score(self, category: str):
        # query
        query = f"""
            -- combine text & image
            with base as (
                select {self.duckdb_cols}
                , 'score_text' match_type
                , score_dense_embed score
                from read_parquet('{self.path_text / f'{category}.parquet'}')
                union all
                select {self.duckdb_cols}
                , 'score_image' match_type
                , score_image_embed score
                from read_parquet('{self.path_img / f'{category}.parquet'}')
            ) 
            -- pivot scores into 2 cols
            , pivot_tab as (
                PIVOT base
                ON match_type
                USING sum(score)
                GROUP BY {self.duckdb_cols}
            )
            -- calculate mean, max rerank
            , cal_tab as (
                select *
                , case 
                    when score_text is null then score_image 
                    when score_image is null then score_text
                    else (score_text + score_image) / 2 end as score_mean
                , greatest(score_text, score_image) as score_max
                from pivot_tab
                where db_item_id != q_item_id
            )
            -- rank matches
            select * 
            , row_number() OVER (PARTITION BY q_item_id ORDER BY score_mean desc) score_rerank
            from cal_tab
            """
        return duckdb.sql(query).pl()

    def run(self):
        for cat in self.all_category:
            df = self.rerank_score(category=cat)
            df = PostProcessing(df).run()
            df.write_parquet(self.path_result / f'{cat}.parquet')

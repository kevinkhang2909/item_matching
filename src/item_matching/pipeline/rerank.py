import polars as pl
import duckdb
from pathlib import Path
from tqdm import tqdm
from rich import print
from core_pro.ultilities import make_dir


class ReRank:
    def __init__(
            self,
            path: Path,
            db_col_idx: str = "db_item_id",
            q_col_idx: str = "q_item_id",
            col_text: str = "item_name"
    ):
        # path
        self.path = path
        self.path_image = self.path / "result_match_image"
        self.path_text = self.path / "result_match_text"
        self.path_result = self.path / "result_match_rerank"
        make_dir(self.path_result)

        # file
        self.file_text = [*self.path_text.glob("*.parquet")]
        self.file_image = [*self.path_image.glob("*.parquet")]

        # all category
        self.all_category = set([i.stem for i in self.file_text] + [i.stem for i in self.file_image])

        # col
        self.col_text = col_text
        self.db_col_idx = db_col_idx
        self.q_col_idx = q_col_idx

    def _data_check(self, category: str):
        paths = {"text": self.path_text, "image": self.path_image}
        df_dict = {
            name: pl.read_parquet(path / f"{category}.parquet")
            for name, path in paths.items()
        }
        return df_dict

    def rerank_score(self, data_text, data_image):
        # query
        query = f"""
            -- combine text & image
            with base as (
                select {self.q_col_idx}
                , q_{self.col_text}
                , {self.db_col_idx}
                , db_{self.col_text}
                , 'score_text' match_type
                , score_text_embed score
                from data_text
                union all
                select {self.q_col_idx}
                , q_{self.col_text}
                , {self.db_col_idx}
                , db_{self.col_text}
                , 'score_image' match_type
                , score_image_embed score
                from data_image
            ) 
            -- pivot scores into 2 cols
            , pivot_tab as (
                PIVOT base
                ON match_type
                USING sum(score)
                GROUP BY {self.q_col_idx}, q_{self.col_text}, {self.db_col_idx}, db_{self.col_text}
            )
            -- calculate mean, max rerank
            , cal_tab as (
                select *
                , (score_text + score_image) / 2 as score_mean
                from pivot_tab
            )
            -- rank matches
            select * exclude(score_mean)
            , coalesce(score_mean, score_text, score_image) score_mean
            from cal_tab
            order by {self.q_col_idx}, coalesce(score_mean, score_text, score_image) desc
            """
        return duckdb.sql(query).pl()

    def run(self):
        total_cat = len(self.all_category)
        for cat in tqdm(self.all_category, total=total_cat, desc="ReRanking"):
            file_name = self.path_result / f"{cat}.parquet"
            if file_name.exists():
                continue
            df_dict = self._data_check(cat)
            df = self.rerank_score(data_text=df_dict["text"], data_image=df_dict["image"])
            if not df.is_empty():
                df.write_parquet(file_name)
        print(f"[RERANK]: Done {total_cat} categories")

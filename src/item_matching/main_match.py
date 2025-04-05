from pathlib import Path
import polars as pl
import duckdb
import re
from time import perf_counter
from core_pro.ultilities import rm_all_folder, make_dir
from .pipeline.build_index_and_query import BuildIndexAndQuery
from .pipeline.data_loading import DataEmbedding
from rich import print


class PipelineMatch:
    def __init__(
        self,
        path: Path,
        PATH_Q: Path,
        PATH_DB: Path,
        MATCH_BY: str = "text",
        COL_CATEGORY: str = "",
        SHARD_SIZE: int = 1_500_000,
        QUERY_SIZE: int = 50_000,
        TOP_K: int = 10,
    ):
        # path
        self.PATH_Q = PATH_Q
        self.PATH_DB = PATH_DB
        self.ROOT_PATH = path
        self.MATCH_BY = MATCH_BY
        self.PATH_RESULT = self.ROOT_PATH / f"result_match_{self.MATCH_BY}"
        make_dir(self.PATH_RESULT)

        # config
        self.COL_CATEGORY = COL_CATEGORY
        self.SHARD_SIZE = SHARD_SIZE
        self.QUERY_SIZE = QUERY_SIZE
        self.TOP_K = TOP_K

        self.lst_category = self._category_chunking()

    def _category_chunking(self) -> list:
        """
        Read by duckdb to perform lazy load
        """
        query = f"""
        select distinct {{1}}{self.COL_CATEGORY} as category 
        from read_parquet('{{0}}') 
        where {{1}}{self.COL_CATEGORY} is not null
        """
        query = query.format(self.PATH_Q, "q_")
        lst_category = duckdb.sql(query).pl()["category"].to_list()
        return sorted(lst_category)

    def _load_data(self, cat: str, mode: str, file: Path):
        """
        Read by polars to prevent special characters in writing query
        """
        filter_ = pl.col(f"{mode}_{self.COL_CATEGORY}") == cat
        return pl.read_parquet(file).filter(filter_)

    def _remove_cache(self):
        folder_list = ["index", "result", "db_array", "db_ds", "q_array", "q_ds", "array", "ds"]
        for name in folder_list:
            rm_all_folder(self.ROOT_PATH / name)

    def run(self):
        # run
        start = perf_counter()
        for idx, cat in enumerate(self.lst_category):
            # logging
            cat_log = f"[dark_orange]{cat}[/]"
            batch_log = f"{idx}/{len(self.lst_category) - 1}"

            # check file exists
            file_result_final = self.PATH_RESULT / f"{cat}.parquet"
            if file_result_final.exists():
                print(f"[PIPELINE] File {cat_log} already exists")
                continue

            # chunk checking
            print("*" * 50)
            print(
                f"🐋 [PIPELINE MATCH BY {self.MATCH_BY}] 🐋 \n"
                f"-> Category: {cat_log} {batch_log}"
            )
            chunk_db = self._load_data(cat=cat, mode="db", file=self.PATH_DB)
            chunk_q = self._load_data(cat=cat, mode="q", file=self.PATH_Q)
            print(
                f"-> Database shape {chunk_db.shape}, Query shape {chunk_q.shape}"
            )

            if chunk_q.shape[0] < 2 or chunk_db.shape[0] < 2:
                print(f"[PIPELINE] Database/Query have not enough data")
                continue

            # embeddings
            DataEmbedding(
                path=self.ROOT_PATH,
                MODE="db",
                MATCH_BY=self.MATCH_BY,
                SHARD_SIZE=self.SHARD_SIZE
            ).load(data=chunk_db)

            DataEmbedding(
                path=self.ROOT_PATH,
                MODE="q",
                MATCH_BY=self.MATCH_BY,
                SHARD_SIZE=self.SHARD_SIZE
            ).load(data=chunk_q)

            # index and query
            cat = re.sub("/", "", cat)  # special characters
            build = BuildIndexAndQuery(
                path=self.ROOT_PATH,
                file_export_name=cat,
                MATCH_BY=self.MATCH_BY,
                QUERY_SIZE=self.QUERY_SIZE,
                TOP_K=self.TOP_K,
            )
            build.build()
            build.query()

        self._remove_cache()

        time_perf = perf_counter() - start
        print(
            f"🐋 [PIPELINE MATCH BY {self.MATCH_BY}] 🐋 \n"
            f"-> Your files are ready, please find here: {self.PATH_RESULT}"
        )
        return {f"time_perf_{self.MATCH_BY}": time_perf}

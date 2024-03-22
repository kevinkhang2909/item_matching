from pathlib import Path
import pandas as pd
import polars as pl
from time import perf_counter
from .build_index.func import clean_text, rm_all_folder, check_file_type, make_dir
from .build_index.matching import BELargeScale


class Matching:
    def __init__(
            self,
            col_category: str,
            path: str | Path | pl.DataFrame | pd.DataFrame,
            file_database: str | Path = None,
            file_query: str | Path = None,
            df_db: pl.DataFrame | pd.DataFrame = None,
            df_q: pl.DataFrame | pd.DataFrame = None,
    ):
        self.path = path
        self.file_database = file_database
        self.file_query = file_query
        self.df_db = df_db
        self.df_q = df_q
        self.col_category = col_category

    def clean_text(self, data: pl.DataFrame, mode: str):
        return (
            data
            .pipe(clean_text)
            .select(pl.all().name.prefix(f'{mode}_'))
            .drop_nulls()
        )

    def run(
            self,
            match_mode: str = 'text',
            export_type: str = 'parquet',
            top_k: int = 10,
    ):
        """
        Run matching processes
        :param match_mode: text | image
        :param use_clean_text: True | False
        :param export_type: parquet | csv
        :param top_k: top k matches
        :return: json
        """
        json_stats = {}

        # check if import df
        if self.file_database:
            self.df_db = check_file_type(self.file_database)
            self.df_q = check_file_type(self.file_query)

        # Database
        if match_mode != 'image':
            self.df_db = self.clean_text(self.df_db, 'db')
            self.df_q = self.clean_text(self.df_q, 'q')

        json_stats.update({'database shape': self.df_db.shape[0]})
        json_stats.update({'query shape': self.df_q.shape[0]})
        print(json_stats['database shape'])
        print(json_stats['query shape'])

        # Match
        be = BELargeScale(self.path, text_sparse=512)
        if match_mode == 'image':
            be = BELargeScale(self.path, img_dim=True)
        elif match_mode == 'text_dense':
            be = BELargeScale(self.path, text_dense=True)

        start = perf_counter()
        path_match_result = self.path / 'result_match'
        make_dir(path_match_result)
        for cat in sorted(self.df_q[f'q_{self.col_category}'].unique()):
            # filter cat
            file_name = path_match_result / f'{cat}.{export_type}'
            chunk_db = self.df_db.filter(pl.col(f'db_{self.col_category}') == cat)
            chunk_q = self.df_q.filter(pl.col(f'q_{self.col_category}') == cat)
            print(f'🐋 Start matching by [{match_mode}] cat: {cat} - Database shape {chunk_db.shape}, Query shape {chunk_q.shape}')

            if chunk_q.shape[0] == 0 or chunk_db.shape[0] == 0:
                print(f'Database/Query have no data')
                continue
            elif file_name.exists():
                print(f'File already exists: {file_name}')
                continue

            # match
            df_match = be.match(chunk_db, chunk_q, top_k=top_k)

            # export
            if export_type == 'parquet':
                df_match.write_parquet(file_name)
            elif export_type == 'csv':
                df_match.write_csv(file_name)

            # remove caches
            rm_all_folder(self.path / 'index')
            rm_all_folder(self.path / 'result')
            for i in ['db', 'q']:
                rm_all_folder(self.path / f'{i}_array')
                rm_all_folder(self.path / f'{i}_ds')

        # update log
        time_perf = perf_counter() - start
        json_stats.update({'time_perf': time_perf})
        json_stats.update({'path result': path_match_result})
        print(f'🐋 Your files are ready, please find here: {self.path}')
        return json_stats

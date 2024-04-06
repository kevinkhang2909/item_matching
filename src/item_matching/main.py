from pathlib import Path
import duckdb
import polars as pl
from time import perf_counter
from .build_index.func import rm_all_folder, clean_text, make_dir
from .build_index.matching import BELargeScale


class Matching:
    def __init__(
            self,
            col_category: str,
            path: Path,
            path_database: Path = None,
            path_query: Path = None,
            query_batch_size: int = 500_000,
            match_mode: str = 'text_dense',
    ):
        self.path = path
        self.path_database = path_database
        self.path_query = path_query
        self.col_category = col_category
        self.query_batch_size = query_batch_size
        self.match_mode = match_mode

    def check_file_type(
            self,
            file_path: Path,
            mode: str = '',
    ) -> dict:
        # check path
        file_type = file_path.suffix[1:]

        # read duckdb
        query = f"""select * from read_{file_type}('{file_path}')"""
        df = duckdb.sql(query).pl()

        # clean data
        if self.match_mode != 'image':
            df = (
                df
                .pipe(clean_text)
                .select(pl.all().name.prefix(f'{mode}_'))
                .drop_nulls()
            )

        # export
        file_path = self.path / f'{mode}_{self.match_mode}_clean.parquet'
        df.write_parquet(file_path)

        # status
        return {
            f'{mode}_file_clean_path': file_path,
            f'{mode}_data_shape': df.shape[0],
            f'{mode}_col_category': sorted(df[f'{mode}_{self.col_category}'].unique())
        }

    def run(
            self,
            export_type: str = 'parquet',
            top_k: int = 10,
    ):
        """
        Run matching processes
        :param export_type: parquet | csv
        :param top_k: top k matches
        :return: json
        """
        # init
        json_stats = {}

        # read file
        status = self.check_file_type(file_path=self.path_database, mode='db')
        json_stats.update(status)
        status = self.check_file_type(file_path=self.path_query, mode='q')
        json_stats.update(status)

        # Match
        be = BELargeScale(self.path, text_sparse=512)
        if self.match_mode == 'image':
            be = BELargeScale(self.path, img_dim=True, query_batch_size=self.query_batch_size)
        elif self.match_mode == 'text_dense':
            be = BELargeScale(self.path, text_dense=True)

        start = perf_counter()
        path_match_result = self.path / 'result_match'
        make_dir(path_match_result)
        for cat in json_stats['q_col_category']:
            # filter cat
            file_name = path_match_result / f'{cat}.{export_type}'

            # read chunk cat
            query = f"""
            select * 
            from read_parquet('{json_stats[f'db_file_clean_path']}') 
            where db_{self.col_category} = '{cat}'
            """
            chunk_db = duckdb.sql(query).pl()
            query = f"""
            select * 
            from read_parquet('{json_stats[f'q_file_clean_path']}') 
            where q_{self.col_category} = '{cat}'
            """
            chunk_q = duckdb.sql(query).pl()
            print(f'🐋 Start matching by [{self.match_mode}] cat: {cat} - Database shape {chunk_db.shape}, Query shape {chunk_q.shape}')

            # check
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
        json_stats.update({'time_perf': time_perf, 'path result': path_match_result})
        print(f'🐋 Your files are ready, please find here: {self.path}')
        return json_stats

from pathlib import Path
import duckdb
import re
from time import perf_counter
from item_matching.func.utilities import rm_all_folder, make_dir
from item_matching.build_index.matching import BELargeScale


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
        # read query file to extract category
        query = f"""select distinct q_{self.col_category} as category from read_parquet('{self.path_query}')"""
        lst_category = duckdb.sql(query).pl()['category'].to_list()

        # Match
        be = BELargeScale(self.path, text_dense=True)
        if self.match_mode == 'image':
            be = BELargeScale(self.path, img_dim=True, query_batch_size=self.query_batch_size)

        path_match_result = self.path / 'result_match'
        make_dir(path_match_result)
        start = perf_counter()
        # Run
        for idx, cat in enumerate(lst_category):
            # read chunk cat
            query = f"""
            select * 
            from read_parquet('{self.path_database}') 
            where db_{self.col_category} = '{cat}'
            """
            chunk_db = duckdb.sql(query).pl()
            query = f"""
            select * 
            from read_parquet('{self.path_query}') 
            where q_{self.col_category} = '{cat}'
            """
            chunk_q = duckdb.sql(query).pl()
            print(f"🐋 Start matching by [{self.match_mode}] cat: {cat} {idx}/{len(lst_category)} - "
                  f"Database shape {chunk_db.shape}, Query shape {chunk_q.shape}")

            # check
            if chunk_q.shape[0] == 0 or chunk_db.shape[0] == 0:
                print(f'Database/Query have no data')
                continue

            cat = re.sub('/', '', cat)
            file_name = path_match_result / f'{cat}.{export_type}'
            if file_name.exists():
                print(f'File already exists: {file_name}')
                continue

            # match
            df_match = be.match(chunk_db, chunk_q, top_k=top_k)

            # export
            if export_type == 'parquet':
                df_match.write_parquet(file_name)
            else:
                df_match.write_csv(file_name)

            # remove caches
            for name in ['index', 'result', 'db_array', 'db_ds', 'q_array', 'q_ds']:
                rm_all_folder(self.path / name)

        # update log
        time_perf = perf_counter() - start
        json_stats = {'time_perf': time_perf, 'path result': path_match_result}
        print(f'🐋 Your files are ready, please find here: {self.path}')
        return json_stats

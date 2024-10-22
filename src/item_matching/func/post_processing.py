import duckdb
from pathlib import Path
from .utilities import make_dir


class PostProcessing:
    def __init__(self, data, verbose: bool = False):
        self.data = data
        self.verbose = verbose
        self.final_col = None
        self.query_add_url_duckdb = ''
        self.query_add_show_img = ''
        self.mode = ['db', 'q']

    @staticmethod
    def select_export_cols(data):
        q_col = sorted(filter(lambda x: 'q' in x and 'image_url' not in x, data.columns))
        db_col = sorted(filter(lambda x: 'db' in x and 'image_url' not in x, data.columns))
        score_col = sorted(filter(lambda x: 'score' in x, data.columns))
        final_col = q_col + db_col + ['q_image_url', 'db_image_url'] + score_col
        return final_col

    def add_url(self):
        for i in self.mode:
            if f'{i}_shop_id' in self.data.columns:
                self.query_add_url_duckdb += f", 'https://shopee.vn/product/' || {i}_shop_id || '/' || {i}_item_id {i}_url\n"
        return self.query_add_url_duckdb

    def add_show_image(self):
        for i in self.mode:
            if f'{i}_image_url' in self.data.columns:
                self.query_add_show_img += f""", '=IMAGE("' || {i}_image_url || '", 1)' {i}_image_url\n"""
        return self.query_add_show_img

    def run(self):
        # init
        show_image = self.add_show_image()
        add_exclude = ''
        if len(show_image) != 0:
            add_exclude += 'EXCLUDE (db_image_url, q_image_url)'
        data = self.data

        # query
        query = f"""
        select * {add_exclude}
        {self.add_show_image()}
        {self.add_url()}
        from data
        """
        if self.verbose:
            print(query)
        else:
            df = duckdb.sql(query).pl()
            select_col = PostProcessing.select_export_cols(df)
            print(select_col)
            df = df.select(select_col)
        return df

    @staticmethod
    def save_parquet_to_csv(path: Path):
        path_export = path.parent / 'result_export'
        make_dir(path_export)
        for f in [*path.glob('*.parquet')]:
            query = f"""COPY (SELECT * FROM read_parquet('{str(f)}')) TO '{path_export}/{f.stem}.csv' (HEADER, DELIMITER ',')"""
            duckdb.sql(query)

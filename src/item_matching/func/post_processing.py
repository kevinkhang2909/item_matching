import duckdb
from pathlib import Path
from .utilities import make_dir


class PostProcessing:
    def __init__(self, data):
        self.data = data
        self.final_col = None
        self.query_add_url_duckdb = ''
        self.query_add_show_img = ''
        self.mode = ['db', 'q']

    def select_export_cols(self):
        q_col = sorted(filter(lambda x: 'q' in x and 'image_url' not in x, self.data.columns))
        db_col = sorted(filter(lambda x: 'db' in x and 'image_url' not in x, self.data.columns))
        score_col = sorted(filter(lambda x: 'score' in x, self.data.columns))
        self.final_col = q_col + db_col + ['q_image_url', 'db_image_url'] + score_col
        return self.final_col

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
        data = self.data
        query = f"""
        select *
        {self.add_show_image()}
        {self.add_url()}
        from data
        """
        df = (
            duckdb.sql(query).pl()
            .select(self.select_export_cols())
        )
        return df


def save_parquet_to_csv(path: Path):
    path_export = path.parent / 'result_export'
    make_dir(path_export)
    for f in [*path.glob('*.parquet')]:
        query = f"""COPY (SELECT * FROM read_parquet('{str(f)}')) TO '{path_export}/{f.stem}.csv' (HEADER, DELIMITER ',')"""
        duckdb.sql(query)

from pathlib import Path
import polars as pl
from build_index.func import clean_text, rm_all_folder
from build_index.matching import BELargeScale


path = Path.home() / 'Downloads/yang'
file = '/home/kevin/Downloads/yang/nonfss (database)/nonfss_itemid_Mom & Baby.csv'
df_db = (
    pl.scan_csv(file)
    .pipe(clean_text)
    .select(pl.all().name.prefix('db_'))
    .collect()
    .drop_nulls()
)

file = '/home/kevin/Downloads/yang/fss (query)/fss_itemid_Mom & Baby.csv'
df_q = (
    pl.scan_csv(file)
    .pipe(clean_text)
    .select(pl.all().name.prefix('q_'))
    .collect()
    .drop_nulls()
)

# match
be = BELargeScale(path, 512)

for cat in sorted(df_q['q_level1_global_be_category'].unique()):
    file_name = path / f'{cat}.parquet'
    chunk_db = df_db.filter(pl.col(f'db_level1_global_be_category') == cat)
    chunk_q = df_q.filter(pl.col(f'q_level1_global_be_category') == cat)
    print(f'Matching cat: {cat}. DB shape {chunk_db.shape}, Q shape {chunk_q.shape}')

    if chunk_q.shape[0] == 0 or chunk_db.shape[0] == 0 or file_name.exists():
        continue

    df_match = be.match(chunk_db, chunk_q, top_k=10)
    df_match.write_parquet(file_name)

    rm_all_folder(path / 'index')
    rm_all_folder(path / 'result')
    for i in ['db', 'q']:
        rm_all_folder(path / f'{i}_array')
        rm_all_folder(path / f'{i}_ds')

    del chunk_db, chunk_q

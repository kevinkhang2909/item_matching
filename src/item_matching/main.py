from pathlib import Path
import pandas as pd
import polars as pl
from .build_index.func import clean_text, rm_all_folder, check_file_type
from .build_index.matching import BELargeScale


def Matching(
        path: [str | Path | pl.DataFrame | pd.DataFrame],
        file_database: [str | Path] = None,
        file_query: [str | Path] = None,
        df_db: [pl.DataFrame | pd.DataFrame] = None,
        df_q: [pl.DataFrame | pd.DataFrame] = None,
):
    # check if import df
    if not df_db:
        df_db = check_file_type(file_database)
        df_q = check_file_type(file_query)

    # Database
    df_db = (
        df_db
        .pipe(clean_text)
        .select(pl.all().name.prefix('db_'))
        .collect()
        .drop_nulls()
    )

    # Query
    df_q = (
        df_q
        .pipe(clean_text)
        .select(pl.all().name.prefix('q_'))
        .collect()
        .drop_nulls()
    )

    # Match
    be = BELargeScale(path, 512)

    for cat in sorted(df_q['q_level1_global_be_category'].unique()):
        file_name = path / f'{cat}.parquet'
        chunk_db = df_db.filter(pl.col(f'db_level1_global_be_category') == cat)
        chunk_q = df_q.filter(pl.col(f'q_level1_global_be_category') == cat)
        print(f'🐋 Start matching cat: {cat} - Database shape {chunk_db.shape}, Query shape {chunk_q.shape}')

        if chunk_q.shape[0] == 0 or chunk_db.shape[0] == 0:
            print(f'Database/Query have no data')
            continue
        elif file_name.exists():
            print(f'File already exists: {file_name}')
            continue

        df_match = be.match(chunk_db, chunk_q, top_k=10)
        df_match.write_parquet(file_name)

        rm_all_folder(path / 'index')
        rm_all_folder(path / 'result')
        for i in ['db', 'q']:
            rm_all_folder(path / f'{i}_array')
            rm_all_folder(path / f'{i}_ds')

        del chunk_db, chunk_q

    print(f'🐋 Your files are ready, please find here: {file_name}')

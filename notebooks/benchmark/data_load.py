from pathlib import Path
import polars as pl
import numpy as np
from FlagEmbedding import BGEM3FlagModel


def load():
    path = Path('/media/kevin/data_4t/Test')
    if str(Path.home()) == '/Users/kevinkhang':
        path = Path.home() / 'Downloads/Data/Item_Matching_Test'
    file = path / 'item_match.parquet'

    col = [
        'q_item_id',
        'q_level1_global_be_category',
        'q_item_name',
        'q_link_first_image',
        'q_item_name_clean',
    ]
    df = (
        pl.read_parquet(file)
        .select(col)
        .unique(subset=col)
        .with_row_index('id')
    )

    print(f'Data Shape: {df.shape}')
    return df, col, path


def embed(df, path):
    item = df['q_item_name_clean'].to_list()
    file_embed = path / 'embed.npy'
    if not file_embed.exists():
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        embeddings = model.encode(
            item,
            batch_size=512,
            max_length=80,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        np.save(file_embed, embeddings)
    else:
        embeddings = np.load(file_embed)

    df = df.with_columns(pl.Series(values=embeddings, name='vector'))
    return embeddings, file_embed, df

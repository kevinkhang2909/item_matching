import polars as pl
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from core_pro.ultilities import make_sync_folder


def load():
    path = make_sync_folder('Item_Matching_Test')
    file = path / 'Description_Images.parquet'

    col = [
        'item_id',
        'level1_global_be_category',
        'item_name',
        'image_url',
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

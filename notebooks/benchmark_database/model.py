from time import perf_counter
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from notebooks.benchmark_database.data_load import load
import polars as pl
from collections import defaultdict
from pathlib import Path
import numpy as np


# init data
df, col, _ = load()


def jina(texts: list):
    start = perf_counter()
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    model.half()
    model.max_seq_length = 80
    task = 'text-matching'
    embeddings = model.encode(
        texts,
        task=task,
        prompt_name=task,
        show_progress_bar=True
    )
    estimated_time = perf_counter() - start
    file_embed = path_array / 'jina.npy'
    np.save(file_embed, embeddings)
    return estimated_time, embeddings.shape


def bge(texts: list):
    start = perf_counter()
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    embeddings = model.encode(
        texts,
        batch_size=8,
        max_length=80,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )['dense_vecs']
    estimated_time = perf_counter() - start
    file_embed = path_array / 'bge.npy'
    np.save(file_embed, embeddings)
    return estimated_time, embeddings.shape


path = Path('/media/kevin/data_4t/Test/benchmark_database')
path_array = path / 'array'

lst = df['q_item_name_clean'].to_list()
dict_ = defaultdict(list)
bench = {
    'jina': jina,
    'bge': bge,
}
for i, f in bench.items():
    time, size = f(lst)
    dict_['name'].append(i)
    dict_['time'].append(time)
    dict_['data_size'].append(size[0])
    dict_['embed_size'].append(size[1])

report = (
    pl.DataFrame(dict_)
    .with_columns((pl.col('data_size') / pl.col('time')).alias('per_sec'))
)
report.write_csv(path / 'model_benchmark.csv')

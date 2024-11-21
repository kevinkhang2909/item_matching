from datasets import Dataset
from time import perf_counter
from autofaiss import build_index
from tqdm import tqdm
import polars as pl
from pymilvus import MilvusClient
import numpy as np
from pathlib import Path
from collections import defaultdict
from data_load import load


# init data
df, col, _ = load()
path = Path('/media/kevin/data_4t/Test/benchmark_database')
embeddings = np.load(path / 'array/bge.npy')
df = df.with_columns(pl.Series(values=embeddings, name='vector'))


def faiss():
    # index
    start = perf_counter()
    file_index = str(path / 'ip.index')

    build_index(
        embeddings=embeddings,
        index_path=file_index,
        index_infos_path=str(path / f'index.json'),
        save_on_disk=True,
        metric_type='ip',
        verbose=30,
    )
    end_index = perf_counter() - start

    # query
    dataset = Dataset.from_polars(df)
    dataset.load_faiss_index('vector', file_index)

    start = perf_counter()
    total_sample = len(dataset)
    batch_size = 20_000
    num_batches = (total_sample + batch_size) // batch_size
    print(f'Total batches: {num_batches}, Batch size: {batch_size:,.0f}')
    for i, idx in tqdm(enumerate(range(num_batches), start=1), total=num_batches):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, total_sample)
        batch = embeddings[start_idx:end_idx]

        score, result = dataset.get_nearest_examples_batch(
            'vector',
            batch,
            k=10,
        )
    end_query = perf_counter() - start
    return end_index, end_query


def milvus():
    data = df.to_dicts()

    # index
    client = MilvusClient("milvus_demo.db")
    collection = 'demo_collection'
    if client.has_collection(collection_name=collection):
        client.drop_collection(collection_name=collection)

    client.create_collection(
        collection_name=collection,
        dimension=embeddings.shape[1],
        metric_type="IP"
    )

    start = perf_counter()
    total_sample = len(data)
    batch_size = 500
    num_batches = (total_sample + batch_size) // batch_size
    print(f'Total batches: {num_batches}, Batch size: {batch_size:,.0f}')
    for i, idx in tqdm(enumerate(range(num_batches), start=1), total=num_batches):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, total_sample)
        batch = data[start_idx:end_idx]
        res = client.insert(collection_name="demo_collection", data=batch)
    end_index = perf_counter() - start

    # query
    start = perf_counter()
    batch_size = 100
    num_batches = (total_sample + batch_size) // batch_size
    print(f'Total batches: {num_batches}, Batch size: {batch_size:,.0f}')
    for i, idx in tqdm(enumerate(range(num_batches), start=1), total=num_batches):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, total_sample)
        batch = embeddings[start_idx:end_idx].tolist()
        res = client.search(
            collection_name=collection,
            data=batch,
            limit=10,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=col
        )
    end_query = perf_counter() - start
    return end_index, end_query


dict_ = defaultdict(list)
bench = {
    'faiss': faiss,
    'milvus': milvus,
}
for i, f in bench.items():
    end_index, end_query = f()
    dict_['name'].append(i)
    dict_['time_index'].append(end_index)
    dict_['time_query'].append(end_query)

report = (
    pl.DataFrame(dict_)
    .with_columns(pl.lit(f'{embeddings.shape}').alias('data_size'))
)
report.write_csv(path / 'database_benchmark.csv')
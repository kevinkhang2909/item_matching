from tqdm import tqdm
from time import perf_counter
from pymilvus import MilvusClient
import polars as pl
import numpy as np
from core_pro.ultilities import make_sync_folder, create_batch_index

# path
path = make_sync_folder('dataset/item_matching')
file = path / 'data_sample_FMCG_clean.parquet'

df = pl.read_parquet(file)
print(df.shape)
print(df.columns)
embeddings = np.random.rand(df.shape[0], 768)
df = df.with_columns(pl.Series('embeddings', embeddings))

client = MilvusClient("milvus_demo.db")
collection = 'demo_collection'
if client.has_collection(collection_name=collection):
    client.drop_collection(collection_name=collection)

client.create_collection(
    collection_name=collection,
    dimension=embeddings.shape[1],
    primary_field_name='item_id',
    metric_type="COSINE",
    enable_dynamic_field=True,
    vector_field_name="embeddings",
)

start = perf_counter()
QUERY_SIZE = 20_000
run = create_batch_index(len(embeddings), QUERY_SIZE)
data = df.to_dicts()
for i, val in run.items():
    start_idx, end_idx = val[0], val[-1]
    batch = data[start_idx:end_idx]
    res = client.insert(collection_name="demo_collection", data=batch)

end = perf_counter() - start
print(f'Total Index Time: {end:,.0f}s')

start = perf_counter()

total_sample = len(data)
batch_size = 10
run = create_batch_index(len(embeddings), QUERY_SIZE)
for i, val in run.items():
    start_idx, end_idx = val[0], val[-1]
    batch = embeddings[start_idx:end_idx].tolist()
    res = client.search(
        collection_name=collection,
        data=batch,
        limit=10,
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=[i for i in df.columns if i != "embeddings"],
    )

end = perf_counter() - start
print(f'Total Query Time: {end:,.0f}s')
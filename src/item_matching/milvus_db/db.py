from tqdm.auto import tqdm
from time import perf_counter
from pymilvus import MilvusClient, Collection
import polars as pl
import numpy as np
from core_pro.ultilities import make_sync_folder, create_batch_index


# path
path = make_sync_folder('dataset/item_matching')
file = path / 'data_sample_FMCG_clean.parquet'

df = pl.read_parquet(file)
print(df.shape)
print(df.columns)
embeddings = np.load(path / "embeddings.npy")

drop_col = ['cluster', 'description', 'images', 'image_url']
df = (
    df.with_columns(pl.Series('embeddings', embeddings))
    .drop(drop_col)
)

client = MilvusClient("milvus_demo.db")
collection_name = 'demo_collection'
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    dimension=embeddings.shape[1],
    primary_field_name='item_id',
    metric_type="COSINE",
    enable_dynamic_field=True,
    vector_field_name="embeddings",
)

start = perf_counter()
QUERY_SIZE = 1000
run = create_batch_index(len(embeddings), QUERY_SIZE)
data = df.to_dicts()
for i, val in tqdm(run.items()):
    start_idx, end_idx = val[0], val[-1]
    batch = data[start_idx:end_idx]
    res = client.insert(collection_name=collection_name, data=batch)

end = perf_counter() - start
print(f'Total Index Time: {end:,.0f}s')

start = perf_counter()
client.load_collection(collection_name=collection_name)
search_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 64  # Higher values give more accurate results but slower performance
    }
}

total_sample = len(data)
batch_size = 100
run = create_batch_index(len(embeddings), QUERY_SIZE)
for i, val in tqdm(run.items()):
    start_idx, end_idx = val[0], val[-1]
    batch = embeddings[start_idx:end_idx].tolist()
    res = client.search(
        collection_name=collection_name,
        data=batch,
        limit=10,
        search_params=search_params,
        output_fields=[i for i in df.columns if i != "embeddings"],
    )

end = perf_counter() - start
print(f'Total Query Time: {end:,.0f}s')
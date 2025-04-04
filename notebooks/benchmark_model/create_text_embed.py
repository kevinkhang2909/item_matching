import duckdb
from core_pro.ultilities import make_sync_folder
from datasets import Dataset
import numpy as np
import polars as pl
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from time import perf_counter


# path
cluster = "FMCG"
path = make_sync_folder("dataset/item_matching")
file = path / f"data_sample_{cluster}_clean.parquet"

# data loading
query = f"""
select item_id
, item_name
from read_parquet('{file}')
limit 10000
"""
df = duckdb.sql(query).pl().unique(["item_id"])
dataset_chunk = Dataset.from_polars(df)
run = df["item_name"].to_list()

result = []
pretrain_name = "bkai-foundation-models/vietnamese-bi-encoder"
file_embed = path / "bi_encode.npy"
model = SentenceTransformer(pretrain_name, model_kwargs={"torch_dtype": "float16"})
start = perf_counter()
embeddings = model.encode(
    run,
    batch_size=512,
    show_progress_bar=True,
    device="cuda",
    normalize_embeddings=True,
    convert_to_numpy=True,
    max_seq_length=80,
)
durations = perf_counter() - start
np.save(file_embed, embeddings.astype(np.float64))
result.append((pretrain_name, durations))

pretrain_name = "BAAI/bge-m3"
file_embed = path / "bge_encode.npy"
model = BGEM3FlagModel(pretrain_name, use_fp16=True)
start = perf_counter()
embeddings = model.encode(
    run,
    batch_size=8,
    max_length=80,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)["dense_vecs"]
np.save(file_embed, embeddings.astype(np.float64))
durations = perf_counter() - start
result.append((pretrain_name, durations))

df_result = pl.DataFrame(result, orient="row", schema=["name", "duration"])
print(df_result)

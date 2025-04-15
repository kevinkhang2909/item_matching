import duckdb
from core_pro.ultilities import make_sync_folder
from datasets import Dataset
import numpy as np
import polars as pl
import torch
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from accelerate import Accelerator
from time import perf_counter

device = Accelerator().device


def setup_bge(texts: list):
    pretrain_name = "BAAI/bge-m3"
    model = BGEM3FlagModel(pretrain_name, use_fp16=True, device=device, normalize_embeddings=True)

    embeddings = model.encode(
        texts,
        batch_size=8,
        max_length=80,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]
    return embeddings


def setup_bge_compress(texts: list):
    pretrain_name = "BAAI/bge-m3"
    static_embedding = StaticEmbedding.from_distillation(pretrain_name, device=device, pca_dims=512)
    model = SentenceTransformer(modules=[static_embedding]).eval()
    model.eval()
    model = torch.compile(model)

    embeddings = model.encode(
        texts,
        batch_size=512,
        show_progress_bar=True,
        device=device,
        normalize_embeddings=True,
        convert_to_numpy=True,
        max_seq_length=80,
    )
    return embeddings


def fast_text_inference(result: list, texts: list, model="bgem3"):
    start = perf_counter()
    if model == "bgem3_compress":
        embeddings = setup_bge_compress(texts)
    else:
        embeddings = setup_bge(texts)
    durations = perf_counter() - start

    file_embed = path / f"{model}_embeds.npy"
    np.save(file_embed, embeddings.astype(np.float32))
    result.append((model, durations))
    return result


# path
cluster = "FMCG"
path = make_sync_folder("dataset/item_matching")
file = path / f"data_sample_{cluster}_clean.parquet"

# data loading
query = f"""
select item_id
, item_name
from read_parquet('{file}')
"""
df = duckdb.sql(query).pl().unique(["item_id"])
dataset_chunk = Dataset.from_polars(df)
texts = df["item_name"].to_list()

# run
result = []
result = fast_text_inference(result=result, texts=texts, model="bgem3")
result = fast_text_inference(result=result, texts=texts, model="bgem3_compress")

# result
df_result = pl.DataFrame(result, orient="row", schema=["name", "duration"])
df_result.write_csv(path / "text_embed_benchmark.csv")
print(df_result)

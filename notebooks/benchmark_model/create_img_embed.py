import duckdb
from core_pro.ultilities import make_sync_folder
from datasets import Dataset
import numpy as np
import polars as pl
from transformers import AutoProcessor, SiglipVisionModel, Dinov2WithRegistersModel
from time import perf_counter
import torch
from accelerate import Accelerator
from PIL import Image
from torch.nn import functional as F

device = Accelerator().device


def run_siglip(result: list, run: list):
    # path
    pretrain_name = "google/siglip-base-patch16-224"
    file_embed = path / "siglip_encode.npy"

    # load model
    img_processor = AutoProcessor.from_pretrained(pretrain_name)
    config = {
        "pretrained_model_name_or_path": pretrain_name,
        "torch_dtype": torch.bfloat16,
    }
    img_model = SiglipVisionModel.from_pretrained(**config).to(device)
    img_model = torch.compile(img_model)

    # inference
    start = perf_counter()
    images = [Image.open(i).convert("RGB") for i in run]
    inputs = img_processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = img_model(**inputs)
    pooled_output = outputs.pooler_output
    norm_embed = F.normalize(pooled_output, p=2, dim=1).cpu().numpy()

    # save
    durations = perf_counter() - start
    np.save(file_embed, norm_embed.astype(np.float64))
    result.append((pretrain_name, durations))
    return result


def run_dinov2(result: list, run: list):
    # path
    pretrain_name = "facebook/dinov2-with-registers-base"
    file_embed = path / "dinov2_encode.npy"

    # load model
    img_processor = AutoProcessor.from_pretrained(pretrain_name)
    config = {
        "pretrained_model_name_or_path": pretrain_name,
        "torch_dtype": torch.bfloat16,
    }
    img_model = Dinov2WithRegistersModel.from_pretrained(**config).to(device)
    img_model = torch.compile(img_model)

    # inference
    start = perf_counter()
    images = [Image.open(i).convert("RGB") for i in run]
    inputs = img_processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = img_model(**inputs)
    pooled_output = outputs.pooler_output
    norm_embed = F.normalize(pooled_output, p=2, dim=1).cpu().numpy()

    # save
    durations = perf_counter() - start
    np.save(file_embed, norm_embed.astype(np.float64))
    result.append((pretrain_name, durations))
    return result


# path
cluster = "FMCG"
path = make_sync_folder("dataset/item_matching")
file = path / f"data_sample_{cluster}_clean.parquet"

# data loading
query = f"""
select item_id
, item_name
, item_name_clean
, level1_global_be_category
, file_path
from read_parquet('{file}')
limit 10000
"""
df = duckdb.sql(query).pl().unique(["item_id"])
dataset_chunk = Dataset.from_polars(df)
run = df["file_path"].to_list()

# run
result = []
result = run_siglip(result, run)
result = run_dinov2(result, run)

# result
df_result = pl.DataFrame(result, orient="row", schema=["name", "duration"])
df_result.write_csv(path / "img_embed_benchmark.csv")
print(df_result)

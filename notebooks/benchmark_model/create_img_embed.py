import duckdb
from datasets import Dataset
import numpy as np
import polars as pl
from transformers import AutoProcessor, SiglipVisionModel, Dinov2WithRegistersModel
from PIL import Image
import torch
from accelerate import Accelerator
from torch.nn import functional as F
from time import perf_counter
from core_pro.ultilities import make_sync_folder

device = Accelerator().device


def process_image(batch, col: str, img_processor, img_model):
    images = [Image.open(i).convert("RGB") for i in batch[col]]
    inputs = img_processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = img_model(**inputs)
    pooled_output = outputs.pooler_output.half()
    norm_embed = F.normalize(pooled_output, p=2, dim=1).cpu().numpy()
    return {"image_embed": norm_embed}


def run_siglip(result: list, dataset_chunk):
    # path
    pretrain_name = "google/siglip-base-patch16-224"
    file_embed = path / "siglip_encode.npy"

    # load model
    img_processor = AutoProcessor.from_pretrained(pretrain_name, use_fast=True)
    config = {
        "pretrained_model_name_or_path": pretrain_name,
        "torch_dtype": torch.bfloat16,
    }
    img_model = SiglipVisionModel.from_pretrained(**config).to(device)
    img_model = torch.compile(img_model)

    # inference
    start = perf_counter()
    dataset_chunk = dataset_chunk.map(
        process_image,
        batch_size=128,
        batched=True,
        fn_kwargs={"col": "file_path", "img_processor": img_processor, "img_model": img_model},
    )
    dataset_chunk.set_format(type="numpy", columns=["image_embed"], output_all_columns=True)

    # save
    durations = perf_counter() - start
    np.save(file_embed, dataset_chunk["image_embed"].astype(np.float64))
    result.append((pretrain_name, durations))
    return result


def run_dinov2(result: list, dataset_chunk):
    # path
    pretrain_name = "facebook/dinov2-with-registers-base"
    file_embed = path / "dinov2_encode.npy"

    # load model
    img_processor = AutoProcessor.from_pretrained(pretrain_name, use_fast=True)
    config = {
        "pretrained_model_name_or_path": pretrain_name,
        "torch_dtype": torch.bfloat16,
    }
    img_model = Dinov2WithRegistersModel.from_pretrained(**config).to(device)
    img_model = torch.compile(img_model)

    # inference
    start = perf_counter()
    dataset_chunk = dataset_chunk.map(
        process_image,
        batch_size=128,
        batched=True,
        fn_kwargs={"col": "file_path", "img_processor": img_processor, "img_model": img_model},
    )
    dataset_chunk.set_format(type="numpy", columns=["image_embed"], output_all_columns=True)

    # save
    durations = perf_counter() - start
    np.save(file_embed, dataset_chunk["image_embed"].astype(np.float64))
    result.append((pretrain_name, durations))
    return result


# path
cluster = "FMCG"
path = make_sync_folder("dataset/item_matching")
file = path / f"data_sample_{cluster}_clean.parquet"

# data loading
query = f"""
select *
from read_parquet('{file}')
limit 10000
"""
df = duckdb.sql(query).pl().unique(["item_id"])
# dataset_chunk = Dataset.from_polars(df)
#
# # run
# result = []
# result = run_siglip(result, dataset_chunk)
# result = run_dinov2(result, dataset_chunk)
#
# # result
# df_result = pl.DataFrame(result, orient="row", schema=["name", "duration"])
# df_result.write_csv(path / "img_embed_benchmark.csv")
# print(df_result)

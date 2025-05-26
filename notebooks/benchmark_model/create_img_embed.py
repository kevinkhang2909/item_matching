import duckdb
from transformers import (
    Siglip2VisionModel,
    AutoProcessor,
    Siglip2VisionConfig,
)
from accelerate import Accelerator
from time import perf_counter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from core_pro.ultilities import make_sync_folder
from tqdm.auto import tqdm
import numpy as np
import sys
from pathlib import Path

sys.path.extend([str(Path.home() / "PycharmProjects/item_matching")])
from src.item_matching.pipeline.data_loading import setup_dinov2, setup_siglip, ImagePathsDataset, collate_batch


device = Accelerator().device
torch.backends.cudnn.benchmark = True


def setup_siglip2():
    pretrain_name = "google/siglip2-base-patch16-224"

    config = Siglip2VisionConfig(
        image_size=224,  # 224/16 = 14 patches per side → 196 total
        patch_size=16,
        num_channels=3,
        embed_dim=768,
        patch_embed_type="conv",  # if SigLIP-2 supports choosing Conv vs Linear
    )
    img_model = Siglip2VisionModel(config)

    img_model = (
        img_model.from_pretrained(
            pretrain_name, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(pretrain_name)
    return torch.compile(img_model), processor


def fast_img_inference(
    result: list,
    file_paths: list[str],
    batch_size: int = 128,
    num_workers: int = 8,
    model: str = "dinov2",
):
    # 1) Load & compile model in mixed precision
    device = torch.device("cuda")
    if model == "siglip":
        img_model = setup_siglip()
    elif model == "siglip2":
        img_model, processor = setup_siglip2()
    else:
        img_model = setup_dinov2()

    # 2) Prepare DataLoader ---
    ds = ImagePathsDataset(file_paths, img_size=224)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    # 3) Inference + save loop
    start = perf_counter()
    list_embeds = []
    with torch.inference_mode():
        for batch in tqdm(loader):
            # async copy to GPU
            batch = batch.to(device, non_blocking=True)
            outputs = img_model(batch)  # bfloat16
            pooled = outputs.pooler_output.half()  # fp16
            normed = F.normalize(pooled, p=2, dim=1)  # still on GPU

            emb = normed.cpu().numpy().astype("float32")  # (B, dim)
            list_embeds.append(emb)

    embeds = np.concatenate(list_embeds, axis=0)
    np.save(path / f"{model}_embeds.npy", embeds)
    durations = perf_counter() - start

    result.append((model, durations))
    return result


# path
cluster = "FMCG"
path = make_sync_folder("dataset/item_matching")
file = path / f"data_sample_{cluster}_clean.parquet"

# data loading
query = f"""
select * exclude(file_path)
, REPLACE(file_path, 'data_4t', '75b198db-809a-4bd2-a97c-e52daa6b3a2d') AS file_path
from read_parquet('{file}')
"""
df = duckdb.sql(query).pl().unique(["item_id"])
file_paths = df["file_path"].to_list()

# run
result = []
# result = fast_img_inference(result=result, file_paths=file_paths, model="siglip2")
result = fast_img_inference(result=result, file_paths=file_paths, model="siglip")
result = fast_img_inference(result=result, file_paths=file_paths, model="dinov2")

# # result
# df_result = pl.DataFrame(result, orient="row", schema=["name", "duration"])
# df_result.write_csv(path / "img_embed_benchmark.csv")
# print(df_result)

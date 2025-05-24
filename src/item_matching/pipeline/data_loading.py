from PIL import Image
import polars as pl
import numpy as np
from pathlib import Path
from rich import print
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from core_pro.ultilities import create_batch_index
from tqdm.auto import tqdm
from FlagEmbedding import BGEM3FlagModel
from transformers import Dinov2WithRegistersModel, Siglip2VisionModel
from .func import _create_folder


device = Accelerator().device


def get_text_model():
    return BGEM3FlagModel(
        "BAAI/bge-m3", use_fp16=True, device=device, normalize_embeddings=True
    )


def text_inference(text_model, save_file_path: Path, iterable_list: list[str]):
    embeddings = text_model.encode(
        iterable_list,
        batch_size=512,
        max_length=50,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]
    np.save(save_file_path, embeddings.astype(np.float32))
    return embeddings


def collate_batch(batch):
    return torch.stack(batch, dim=0)


class ImagePathsDataset(Dataset):
    def __init__(self, file_paths: list, img_size: int = 224):
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [
                # transforms.Resize(
                #     img_size, interpolation=transforms.InterpolationMode.BICUBIC
                # ),
                # transforms.CenterCrop(img_size),
                transforms.ConvertImageDtype(torch.float32),  # to [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        tensor = transforms.ToTensor()(img)  # HWC→CHW float32
        return self.transform(tensor)


def get_img_model():
    pretrain_name = "google/siglip2-base-patch16-224"
    img_model = (
        Siglip2VisionModel.from_pretrained(
            pretrain_name,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    # pretrain_name = "facebook/dinov2-with-registers-base"
    # img_model = (
    #     Dinov2WithRegistersModel.from_pretrained(
    #         pretrain_name,
    #         torch_dtype=torch.bfloat16,
    #     )
    #     .to(device)
    #     .eval()
    # )
    # return torch.compile(img_model)
    return img_model


def img_inference(
    img_model,
    save_file_path: Path,
    iterable_list: list[str],
    batch_size: int = 128,
    num_workers: int = 8,
):
    # 1) Prepare DataLoader ---
    ds = ImagePathsDataset(iterable_list, img_size=224)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    # 2) Inference + collect embeddings
    all_embs = []
    with torch.inference_mode():
        for batch in tqdm(loader):
            # async copy to GPU
            batch = batch.to(device, non_blocking=True)
            outputs = img_model(batch)  # bfloat16
            pooled = outputs.pooler_output.half()  # fp16
            normed = F.normalize(pooled, p=2, dim=1)  # still on GPU

            emb = normed.cpu().numpy().astype("float32")  # (B, dim)
            all_embs.append(emb)

    embeddings = np.concatenate(all_embs, axis=0)
    np.save(save_file_path, embeddings)
    return embeddings


class DataEmbedding:
    def __init__(
        self,
        path: Path,
        MODE: str,
        MATCH_BY: str = "text",
        SHARD_SIZE: int = 1_500_000,
    ):
        # Config
        self.MATCH_BY = MATCH_BY
        self.MODE = MODE
        self.SHARD_SIZE = SHARD_SIZE

        # Path
        self.path = path
        dict_array_path = _create_folder(path, "array")
        dict_ds_path = _create_folder(path, "ds")
        self.path_array = dict_array_path[self.MODE]
        self.path_ds = dict_ds_path[self.MODE]

        # Model
        self._prepare_col_input_model()

    def _prepare_col_input_model(self):
        if self.MATCH_BY == "text":
            self.col_input = f"{self.MODE}_item_name_clean"
            self.col_embedding = f"{self.MATCH_BY}_embed"
            self.text_model = get_text_model()
        else:
            self.col_input = f"{self.MODE}_file_path"
            self.col_embedding = f"{self.MATCH_BY}_embed"
            self.img_model = get_img_model()

    def load(self, data: pl.DataFrame):
        # Log total chunks
        run = create_batch_index(data.shape[0], self.SHARD_SIZE)
        num_chunks = len(run)

        # Process and save each chunk
        for i, idx in run.items():
            # Check if exists:
            dataset_name = self.path_ds / f"{i}.parquet"
            array_name = self.path_array / f"{i}.npy"
            if dataset_name.exists():
                continue

            # Load Chunk
            start_idx, end_idx = idx[0], idx[-1]
            if start_idx == end_idx:  # prevent sample size is 1
                end_idx = None

            dataset_chunk = data[start_idx:end_idx]
            print(
                f"[DataEmbedding] Shard [{i}/{num_chunks - 1}]: start {start_idx:,.0f} end {end_idx:,.0f}"
            )

            # Process dataset
            if self.MATCH_BY == "text":
                embeddings = text_inference(
                    text_model=self.text_model,
                    save_file_path=array_name,
                    iterable_list=dataset_chunk[self.col_input].to_list(),
                )
            else:
                embeddings = img_inference(
                    img_model=self.img_model,
                    save_file_path=array_name,
                    iterable_list=dataset_chunk[self.col_input].to_list(),
                )

            # Concat
            dset_embed = pl.DataFrame({self.col_embedding: embeddings})
            dataset_chunk = pl.concat([dataset_chunk, dset_embed], how="horizontal")

            # Save chunk
            dataset_chunk.write_parquet(str(dataset_name))

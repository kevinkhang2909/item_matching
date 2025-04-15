import duckdb
import polars as pl
from transformers import SiglipVisionModel, Dinov2WithRegistersModel
from PIL import Image
from accelerate import Accelerator
from time import perf_counter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from numpy.lib.format import open_memmap
from core_pro.ultilities import make_sync_folder
from tqdm.auto import tqdm

device = Accelerator().device
torch.backends.cudnn.benchmark = True


class ImagePathsDataset(Dataset):
    def __init__(self, file_paths: list, img_size: int = 224):
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(img_size),
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


def collate_batch(batch):
    return torch.stack(batch, dim=0)


def setup_siglip():
    pretrain_name = "google/siglip-base-patch16-224"
    img_model = (
        SiglipVisionModel.from_pretrained(
            pretrain_name,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    return torch.compile(img_model)


def setup_dinov2():
    pretrain_name = "facebook/dinov2-with-registers-base"
    img_model = (
        Dinov2WithRegistersModel.from_pretrained(
            pretrain_name,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    return torch.compile(img_model)


def fast_img_inference(
    result: list,
    file_paths: list[str],
    batch_size: int = 128,
    num_workers: int = 8,
    model: str = "dinov2",
):
    # 1) Prepare DataLoader ---
    ds = ImagePathsDataset(file_paths, img_size=224)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    # 2) Load & compile model in mixed precision
    device = torch.device("cuda")
    if model == "siglip":
        img_model = setup_siglip()
    else:
        img_model = setup_dinov2()

    # 3) Pre‑allocate a .npy memmap for all embeddings
    total = len(ds)
    dim = img_model.config.hidden_size  # e.g. 1024
    mmap = open_memmap(
        filename=str(path / f"{model}_embeds.npy"),
        mode="w+",
        dtype="float32",
        shape=(total, dim),
    )

    # 4) Inference + save loop
    start = perf_counter()
    idx = 0
    with torch.inference_mode():
        for batch in tqdm(loader):
            # async copy to GPU
            batch = batch.to(device, non_blocking=True)
            outputs = img_model(batch)  # bfloat16
            pooled = outputs.pooler_output.half()  # fp16
            normed = F.normalize(pooled, p=2, dim=1)  # still on GPU

            emb = normed.cpu().numpy().astype("float32")  # (B, dim)
            bs = emb.shape[0]
            mmap[idx : idx + bs] = emb  # write into .npy
            idx += bs

    durations = perf_counter() - start
    mmap.flush()  # ensure all data is on disk

    result.append((model, durations))
    return result


# path
cluster = "FMCG"
path = make_sync_folder("dataset/item_matching")
file = path / f"data_sample_{cluster}_clean.parquet"

# data loading
query = f"""
select *
from read_parquet('{file}')
"""
df = duckdb.sql(query).pl().unique(["item_id"])
file_paths = df["file_path"].to_list()

# run
result = []
result = fast_img_inference(result=result, file_paths=file_paths, model="siglip")
result = fast_img_inference(result=result, file_paths=file_paths, model="dinov2")

# result
df_result = pl.DataFrame(result, orient="row", schema=["name", "duration"])
df_result.write_csv(path / "img_embed_benchmark.csv")
print(df_result)

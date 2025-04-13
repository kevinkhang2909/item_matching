import polars as pl
from datasets import Dataset, concatenate_datasets
from pathlib import Path
import numpy as np
from rich import print
from core_pro.ultilities import create_batch_index, make_dir
from ..model.model import Model
from .func import _create_folder


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
        self.model = Model()
        if self.MATCH_BY == "text":
            self.col_input = f"{self.MODE}_item_name_clean"
            self.col_embedding = f"{self.MATCH_BY}_embed"
            self.model.get_text_model()
        else:
            self.col_input = f"{self.MODE}_file_path"
            self.model.get_img_model()
            self.col_embedding = f"{self.MATCH_BY}_embed"

    def load(self, data: pl.DataFrame):
        # Log total chunks
        run = create_batch_index(data.shape[0], self.SHARD_SIZE)
        num_chunks = len(run)

        # Process and save each chunk
        for i, idx in run.items():
            # Check if exists:
            dataset_name = self.path_ds / f"{i}"
            array_name = self.path_array / f"{i}.npy"
            if dataset_name.exists():
                continue

            # Load Chunk
            start_idx, end_idx = idx[0], idx[-1]
            if start_idx == end_idx:  # prevent sample size is 1
                end_idx = None

            dataset_chunk = Dataset.from_polars(data[start_idx:end_idx])
            print(
                f"[DataEmbedding] Shard [{i}/{num_chunks - 1}]: start {start_idx:,.0f} end {end_idx:,.0f}"
            )

            # Process dataset
            if self.MATCH_BY == "text":
                embeddings = self.model.process_text(dataset_chunk[self.col_input])
                dset_embed = Dataset.from_dict({self.col_embedding: embeddings})
                dataset_chunk = concatenate_datasets([dataset_chunk, dset_embed], axis=1)
            else:
                dataset_chunk = dataset_chunk.map(
                    self.model.process_image,
                    batch_size=512,
                    batched=True,
                    fn_kwargs={"col": self.col_input},
                )

            # Normalize
            dataset_chunk.set_format(type="torch", columns=[self.col_embedding], output_all_columns=True)
            if self.MATCH_BY == "image":
                dataset_chunk = dataset_chunk.map(
                    Model.pp_normalize,
                    batched=True,
                    fn_kwargs={"col": self.col_embedding},
                )
            dataset_chunk.set_format(
                type="numpy", columns=[self.col_embedding], output_all_columns=True
            )

            # Save chunk
            np.save(array_name, dataset_chunk[self.col_embedding])
            dataset_chunk.save_to_disk(str(dataset_name))

from pydantic import BaseModel, Field, computed_field
import polars as pl
from datasets import Dataset, concatenate_datasets
from pathlib import Path
import numpy as np
from rich import print
from core_pro.ultilities import create_batch_index, make_dir
from ..model.model import Model


class ConfigEmbedding(BaseModel):
    ROOT_PATH: Path = Field(default=None)
    SHARD_SIZE: int = Field(default=1_500_000)
    MODE: str = Field(default='')
    MATCH_BY: str = Field(default='text')

    @computed_field
    @property
    def col_embedding(self) -> str:
        dict_ = {'image': 'image_embed'}
        return dict_.get(self.MATCH_BY, 'text_embed')

    @computed_field
    @property
    def col_input(self) -> str:
        col = 'file_path' if self.MODE == '' else f'{self.MODE}_file_path'
        default_val = 'item_name_clean' if self.MODE == '' else f'{self.MODE}_item_name_clean'
        dict_ = {'image': col}
        return dict_.get(self.MATCH_BY, default_val)

    @computed_field
    @property
    def path_array(self) -> Path:
        folder = 'array' if self.MODE == '' else f'{self.MODE}_array'
        return self.ROOT_PATH / folder

    @computed_field
    @property
    def path_ds(self) -> Path:
        folder = 'ds' if self.MODE == '' else f'{self.MODE}_ds'
        return self.ROOT_PATH / folder


class DataEmbedding:
    def __init__(self, config_input: ConfigEmbedding):
        # Path
        self.MATCH_BY = config_input.MATCH_BY
        self.SHARD_SIZE = config_input.SHARD_SIZE
        self.col_embedding = config_input.col_embedding
        self.col_input = config_input.col_input
        self.path_array = config_input.path_array
        self.path_ds = config_input.path_ds

        # Init path
        make_dir(self.path_array)
        make_dir(self.path_ds)

    def load(self, data: pl.DataFrame):
        # Log total chunks
        run = create_batch_index(data.shape[0], self.SHARD_SIZE)
        num_chunks = len(run)

        # Model
        model = Model()
        if self.MATCH_BY == 'text':
            model.get_text_model()
        elif self.MATCH_BY == 'image':
            model.get_img_model()

        # Process and save each chunk
        for i, idx in run.items():
            # Check if exists:
            dataset_name = self.path_ds / f'{i}'
            array_name = self.path_array / f'{i}.npy'
            if dataset_name.exists():
                continue

            # Load Chunk
            start_idx, end_idx = idx[0], idx[-1]
            dataset_chunk = Dataset.from_polars(data[start_idx:end_idx])
            print(f'[DataEmbedding] Shard [{i}/{num_chunks - 1}]: start {start_idx:,.0f} end {end_idx:,.0f}')

            # Process dataset
            if self.MATCH_BY == 'text':
                embeddings = model.process_text(dataset_chunk[self.col_input])
                dset_embed = Dataset.from_dict({self.col_embedding: embeddings})
                dataset_chunk = concatenate_datasets([dataset_chunk, dset_embed], axis=1)
            else:
                dataset_chunk = dataset_chunk.map(
                    model.process_image,
                    batch_size=512,
                    batched=True,
                    fn_kwargs={'col': self.col_input}
                )

            # Normalize
            dataset_chunk.set_format(type='torch', columns=[self.col_embedding], output_all_columns=True)
            if self.MATCH_BY == 'image':
                dataset_chunk = dataset_chunk.map(Model.pp_normalize, batched=True, fn_kwargs={'col': self.col_embedding})
            dataset_chunk.set_format(type='numpy', columns=[self.col_embedding], output_all_columns=True)

            # Save chunk
            np.save(array_name, dataset_chunk[self.col_embedding])
            dataset_chunk.save_to_disk(str(dataset_name))

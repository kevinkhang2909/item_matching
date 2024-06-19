from pydantic import BaseModel, Field, computed_field
import polars as pl
from datasets import Dataset, concatenate_datasets
from pathlib import Path
import numpy as np
from item_matching.model.model import Model
from item_matching.func.utilities import make_dir
from loguru import logger


class ConfigEmbedding(BaseModel):
    ROOT_PATH: Path = Field(default=None)
    SHARD_SIZE: int = Field(default=1_500_000)
    MODE: str = Field(default='')
    MATCH_BY: str = Field(default='text')

    @computed_field
    @property
    def col_embedding(self) -> str:
        dict_ = {'image': 'image_embed'}
        return dict_.get(self.MATCH_BY, 'dense_embed')

    @computed_field
    @property
    def col_input(self) -> str:
        dict_ = {'image': f'{self.MODE}_file_path'}
        return dict_.get(self.MATCH_BY, f'{self.MODE}_item_name_clean')

    @computed_field
    @property
    def path_array(self) -> Path:
        return self.ROOT_PATH / f'{self.MODE}_array'

    @computed_field
    @property
    def path_ds(self) -> Path:
        return self.ROOT_PATH / f'{self.MODE}_ds'


class DataEmbedding:
    def __init__(self, config_input: ConfigEmbedding):
        # Path
        self.config_input = config_input

        # Init path
        make_dir(self.config_input.path_array)
        make_dir(self.config_input.path_ds)

    def load(self, data: pl.DataFrame):
        # Log total chunks
        total_sample = len(data)
        num_chunks = (total_sample + self.config_input.SHARD_SIZE) // self.config_input.SHARD_SIZE

        # Model
        model = Model()
        if self.config_input.MATCH_BY == 'text':
            model.get_text_model()
        elif self.config_input.MATCH_BY == 'image':
            model.get_img_model()
        col_embed = self.config_input.col_embedding

        # Process and save each chunk
        logger.info(
            f'[DataEmbedding] Total chunks {self.config_input.MODE}: {num_chunks} '
            f'- Shard size: {self.config_input.SHARD_SIZE:,.0f}'
        )
        for i, idx in enumerate(range(num_chunks), start=1):
            # Load Chunk
            start_idx = idx * self.config_input.SHARD_SIZE
            end_idx = min(start_idx + self.config_input.SHARD_SIZE, total_sample)
            dataset_chunk = Dataset.from_polars(data[start_idx:end_idx])
            logger.info(f'Shard [{i}/{num_chunks}]: start {start_idx:,.0f} end {end_idx:,.0f}')

            # Process dataset
            if self.config_input.MATCH_BY == 'text':
                embeddings = model.process_text(dataset_chunk[self.config_input.col_input])
                dset_embed = Dataset.from_dict({col_embed: embeddings})
                dataset_chunk = concatenate_datasets([dataset_chunk, dset_embed], axis=1)
            else:
                dataset_chunk.map(model.process_text, batched=True, fn_kwargs={'col': self.config_input.col_input})

            # Normalize
            dataset_chunk.set_format(type='torch', columns=[col_embed], output_all_columns=True)
            dataset_chunk.map(Model.pp_normalize, batched=True, fn_kwargs={'col': col_embed})
            dataset_chunk.set_format(type='numpy', columns=[col_embed], output_all_columns=True)

            # Save chunk
            np.save(self.config_input.path_array / f'{idx}.npy', dataset_chunk[col_embed])
            dataset_chunk.save_to_disk(str(self.config_input.path_ds / f'{idx}'))

from pathlib import Path
import polars as pl
from time import perf_counter
from re import search
from autofaiss import build_index
from datasets import concatenate_datasets, load_from_disk
from pydantic import BaseModel, Field, computed_field
import numpy as np
from rich import print
from item_matching.func.utilities import make_dir
from core_pro.ultilities import create_batch_index


class ConfigQuery(BaseModel):
    ROOT_PATH: Path = Field(default=None)
    QUERY_SIZE: int = Field(default=50_000)
    MATCH_BY: str = Field(default='text')
    TOP_K: int = Field(default=10)

    @computed_field
    @property
    def col_embedding(self) -> str:
        dict_ = {'image': 'image_embed'}
        return dict_.get(self.MATCH_BY, 'text_embed')

    @computed_field
    @property
    def path_array_db(self) -> Path:
        return self.ROOT_PATH / f'db_array'

    @computed_field
    @property
    def path_ds_db(self) -> Path:
        return self.ROOT_PATH / f'db_ds'

    @computed_field
    @property
    def path_ds_q(self) -> Path:
        return self.ROOT_PATH / f'q_ds'

    @computed_field
    @property
    def path_ds_inner(self) -> Path:
        return self.ROOT_PATH / f'ds'

    @computed_field
    @property
    def path_array_inner(self) -> Path:
        return self.ROOT_PATH / f'array'

    @computed_field
    @property
    def path_index(self) -> Path:
        return self.ROOT_PATH / f'index'

    @computed_field
    @property
    def path_result(self) -> Path:
        path_result = self.ROOT_PATH / f'result_match_{self.MATCH_BY}'
        make_dir(path_result)
        return path_result


class BuildIndexAndQuery:
    def __init__(
            self,
            file_export_name: str,
            config: ConfigQuery,
            inner: bool = False,
            explode: bool = True
    ):
        self.TOP_K = config.TOP_K
        self.QUERY_SIZE = config.QUERY_SIZE
        self.col_embedding = config.col_embedding
        self.inner = inner
        self.explode = explode
        self.sort_key_ds = lambda x: int(x.stem)
        self.sort_key_result = lambda x: int(x.stem.split('_')[1])

        # index
        self.path_index = config.path_index
        self.file_index = self.path_index / f'ip.index'
        self.file_index_json = str(self.path_index / f'index.json')

        # array
        self.path_array_db = config.path_array_inner if inner else config.path_array_db

        # ds
        self.dataset_dict = {
            'db_ds_path': config.path_ds_db,
            'q_ds_path': config.path_ds_q,
            'inner_ds_path': config.path_ds_inner,
        }

        # result
        self.path_result = config.path_result
        self.file_result = self.path_result / f'{file_export_name}.parquet'

    def build(self):
        # Build index
        start = perf_counter()
        if not self.file_index.exists():
            build_index(
                str(self.path_array_db),
                index_path=str(self.file_index),
                index_infos_path=self.file_index_json,
                save_on_disk=True,
                metric_type='ip',
                verbose=30,
            )
            print(f'[BuildIndex] Time finished: {perf_counter() - start:,.2f}s')
        else:
            print(f'[BuildIndex] Index is existed')

    def load_dataset(self):
        dataset = {}
        if self.inner:
            for i in ['db', 'q']:
                dataset[i] = concatenate_datasets([
                    load_from_disk(str(f))
                    for f in sorted(self.dataset_dict[f'inner_ds_path'].glob('*'), key=self.sort_key_ds)
                ])

                for c in dataset[i].column_names:
                    if c != self.col_embedding:
                        dataset[i] = dataset[i].rename_column(i, f'{i}_{c}')

        else:
            for i in ['db', 'q']:
                dataset[i] = concatenate_datasets([
                    load_from_disk(str(f))
                    for f in sorted(self.dataset_dict[f'{i}_ds_path'].glob('*'), key=self.sort_key_ds)
                ])

        # Add index
        dataset['db'].load_faiss_index(self.col_embedding, self.file_index)
        return dataset['db'], dataset['q']

    def query(self):
        # Check file exists
        if not self.file_result.exists():
            # Load dataset
            dataset_db, dataset_q = self.load_dataset()

            # Batch query
            run = create_batch_index(len(dataset_q), self.QUERY_SIZE)
            num_batches = len(run)
            for i, val in run.items():
                # init
                file_name_result = self.path_result / f'result_{i}.parquet'
                file_name_score = self.path_result / f'score_{i}.parquet'
                if file_name_result.exists():
                    continue

                # query
                start_idx, end_idx = val[0], val[-1]
                start_batch = perf_counter()
                score, result = dataset_db.get_nearest_examples_batch(
                    self.col_embedding,
                    dataset_q[start_idx:end_idx][self.col_embedding],
                    k=self.TOP_K,
                )
                # export
                for arr in result:
                    del arr[self.col_embedding]  # prevent memory leaks
                df_result = pl.DataFrame(result)
                df_result.write_parquet(file_name_result)

                # track errors
                if df_result.shape[0] == 0:
                    print(f"[red]No matches found for {i}[/]")
                    return pl.DataFrame()

                dict_ = {f'score_{self.col_embedding}': [list(np.round(arr, 6)) for arr in score]}
                df_score = pl.DataFrame(dict_)
                df_score.write_parquet(file_name_score)

                # log
                end = perf_counter() - start_batch
                print(f"[Query] Batch {i}/{num_batches} match result shape: {df_result.shape} {end:,.2f}s")
                del score, result, df_score, df_result

            # Post process
            dataset_q = dataset_q.remove_columns(self.col_embedding)  # prevent polars issues
            del dataset_db

            df_score = (
                pl.concat([
                    pl.read_parquet(f)
                    for f in sorted(self.path_result.glob('score*.parquet'), key=self.sort_key_result)])
            )
            df_result = (
                pl.concat([
                    pl.read_parquet(f)
                    for f in sorted(self.path_result.glob('result*.parquet'), key=self.sort_key_result)])
            )

            df_match = pl.concat([dataset_q.to_polars(), df_result, df_score], how='horizontal')
            if self.explode:
                col_explode = [i for i in df_match.columns if search('db|score', i)]
                df_match = df_match.explode(col_explode)

            df_match.write_parquet(self.file_result)
        else:
            print(f'[Match] File {self.file_result.stem} already exists')
        return self.file_result

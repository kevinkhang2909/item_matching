from pathlib import Path
import polars as pl
from time import perf_counter
from autofaiss import build_index
from datasets import concatenate_datasets, load_from_disk
import numpy as np
from rich import print
from core_pro.ultilities import create_batch_index, make_dir
from .func import _create_folder, _round_score
from re import search


class BuildIndexAndQuery:
    def __init__(
        self,
        path: Path,
        file_export_name: str,
        MATCH_BY: str = "text",
        TOP_K: int = 10,
        QUERY_SIZE: int = 50_000,
    ):
        self.path = path
        self.TOP_K = TOP_K
        self.QUERY_SIZE = QUERY_SIZE
        self.MATCH_BY = MATCH_BY
        self.file_export_name = file_export_name
        self._prepare_col_input_model()

        # index
        self.path_index = _create_folder(path, "index", one=True)
        self.file_index = self.path_index / f"ip.index"
        self.file_index_json = str(self.path_index / f"index.json")

        # array
        self.path_array_db = path / "db_array"

        # ds
        self.dataset_dict = {f"{i}_ds_path": self.path / f"{i}_ds" for i in ["db", "q"]}

        # sort key
        self.sort_key_ds = lambda x: int(x.stem)
        self.sort_key_result = lambda x: int(x.stem.split("_")[1])

        # result
        self._create_folder_result()

    def _create_folder_result(self):
        self.path_result_query_score = self.path / f"result"
        self.path_result_final = self.path / f"result_match_{self.MATCH_BY}"
        make_dir(self.path_result_query_score)
        make_dir(self.path_result_final)
        self.file_export_final = self.path_result_final / f"{self.file_export_name}.parquet"

    def _prepare_col_input_model(self):
        if self.MATCH_BY == "text":
            self.col_embedding = f"{self.MATCH_BY}_embed"
        else:
            self.col_embedding = f"{self.MATCH_BY}_embed"

    def build(self):
        # Build index
        start = perf_counter()
        if not self.file_index.exists():
            print(f"[BuildIndex] Start")
            try:
                build_index(
                    str(self.path_array_db),
                    index_path=str(self.file_index),
                    index_infos_path=str(self.file_index_json),
                    save_on_disk=True,
                    metric_type="ip",
                    verbose=30,
                )
                print(f"[BuildIndex] Time finished: {perf_counter() - start:,.2f}s")
            except TypeError as e:
                print(f"[BuildIndex] Error: {e}")
                return None
        else:
            print(f"[BuildIndex] Index is existed")

    def load_dataset(self):
        dataset = {}
        for i in ["db", "q"]:
            files = sorted(self.dataset_dict[f"{i}_ds_path"].glob("*"), key=self.sort_key_ds)
            lst_ds = [load_from_disk(str(f)) for f in files]
            dataset[i] = concatenate_datasets(lst_ds)

        # Add index
        dataset["db"].load_faiss_index(self.col_embedding, self.file_index)
        return dataset["db"], dataset["q"]

    def query(self):
        # Load dataset
        dataset_db, dataset_q = self.load_dataset()

        # Batch query
        run = create_batch_index(len(dataset_q), self.QUERY_SIZE)
        num_batches = len(run)
        for i, val in run.items():
            # init
            file_name_result = self.path_result_query_score / f"result_{i}.parquet"
            file_name_score = self.path_result_query_score / f"score_{i}.parquet"
            if file_name_result.exists():
                continue

            # query
            start_idx, end_idx = val[0], val[-1]
            if start_idx == end_idx:  # prevent sample size is 1
                end_idx = None

            start_batch = perf_counter()
            score, result = dataset_db.get_nearest_examples_batch(
                self.col_embedding,
                np.asarray(dataset_q[start_idx:end_idx][self.col_embedding]),
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
                continue

            dict_ = {f"score_{self.col_embedding}": [_round_score(arr) for arr in score]}
            df_score = pl.DataFrame(dict_)
            df_score.write_parquet(file_name_score)

            # log
            end = perf_counter() - start_batch
            print(
                f"[Query] Batch {i}/{num_batches - 1} match result shape: {df_result.shape} {end:,.2f}s"
            )
            del score, result, df_score, df_result

        # Concat all files
        dataset_q = dataset_q.remove_columns(self.col_embedding)  # prevent polars issues
        del dataset_db

        # score
        files_score = sorted(self.path_result_query_score.glob("score*.parquet"), key=self.sort_key_result)
        df_score = pl.concat([pl.read_parquet(f) for f in files_score])

        # result
        files_result = sorted(self.path_result_query_score.glob("result*.parquet"), key=self.sort_key_result)
        df_result = pl.concat([pl.read_parquet(f) for f in files_result])

        # combine to data
        df_match = pl.concat([dataset_q.to_polars(), df_result, df_score], how="horizontal")

        # explode result
        col_explode = [i for i in df_match.columns if search("db|score", i)]
        df_match = df_match.explode(col_explode)

        # export
        df_match.write_parquet(self.file_export_final)

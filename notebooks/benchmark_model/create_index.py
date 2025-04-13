from pathlib import Path
import polars as pl
from datasets import Dataset
import numpy as np
from autofaiss import build_index



def create_search_result(path: Path, file_embed: Path, data: pl.DataFrame):
    embeddings = np.load(file_embed)
    print(embeddings.shape)

    data = data.with_columns(pl.Series(values=embeddings, name="embed"))
    dataset = Dataset.from_polars(data)
    dataset.set_format(type="numpy", columns=["embed"], output_all_columns=True)

    path_index = Path(path / f"index_{file_embed.stem}")
    build_index(
        embeddings=embeddings,
        index_path=str(path_index),
        save_on_disk=True,
        metric_type="ip",
        verbose=30,
    )
    dataset.load_faiss_index("embed", path_index)

    score, result = dataset.get_nearest_examples_batch(
        "embed", dataset["embed"], k=5
    )
    for i in result:
        del i["embed"]

    dict_ = {"score": [list(i) for i in score]}
    df_score = pl.DataFrame(dict_)
    df_result = (
        pl.DataFrame(result).select(pl.all().name.prefix(f"db_"))
    )
    df_match = pl.concat([data, df_result, df_score], how="horizontal")
    return df_match

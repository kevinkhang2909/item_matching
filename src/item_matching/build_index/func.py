from pathlib import Path
import polars as pl


def clean_text(data, col: str = 'item_name'):
    import re

    regex = "[\(\[\<\"].*?[\)\]\>\"]"
    return data.with_columns(
        pl.col(col).map_elements(
            lambda x: re.sub(regex, "", x).lower().rstrip('.').strip()
        )
        .alias(f'{col.lower()}_clean')
    )


def ngrams_func(string):
    ngrams = zip(*[string[i:] for i in range(3)])
    return [''.join(ngram) for ngram in ngrams]


def tfidf(lst_item, dim: int = 512):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer=ngrams_func, max_features=dim)
    vectorizer.fit(lst_item)
    return vectorizer


def rm_all_folder(path: Path):
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_all_folder(child)
    path.rmdir()


def make_dir(folder_name):
    if not folder_name.exists():
        folder_name.mkdir(parents=True, exist_ok=True)


def check_file_type(file_path):
    file_type = ''
    if isinstance(file_path, str):
        file_type = Path(file_path).suffix[1:]

    dict_ = {
        'csv': pl.read_csv,
        'parquet': pl.read_parquet
    }
    df = dict_[file_type](file_path)
    return df


def download_images(path: Path, file_name: str, col_url: str):
    import os
    import subprocess

    os.chdir(str(path))
    command = (
        f"img2dataset --url_list={file_name}_0.parquet "
        f"--output_folder=img_{file_name}/ "
        f"--processes_count=16 "
        f"--thread_count=32 "
        f"--image_size=224 "
        f"--output_format=files "
        f"--input_format=parquet "
        f"--url_col={col_url} "
        f"--number_sample_per_shard=50000 "
    )
    subprocess.run(command, shell=True)


def load_images(path: Path, mode: str = '', col_url: str = 'url') -> pl.DataFrame:
    import json

    # listing
    lst_json = [str(i) for i in sorted(path.glob('*/*.json'))]
    lst_img = [str(i) for i in sorted(path.glob('*/*.jpg'))]
    lst_file = [json.loads(open(i, "r").read())['url'] for i in lst_json]
    return pl.DataFrame({
        f'{mode}_{col_url}': lst_file,
        f'{mode}_file_path': lst_img,
        f'{mode}_exists': [True for _ in lst_file],
    })

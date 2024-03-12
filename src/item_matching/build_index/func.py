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
    if isinstance(file_path, str):
        file_type = Path(file_path).suffix[1:]
    else:
        file_type = file_path.suffix[1:]

    dict_ = {
        'csv': pl.read_csv,
        'parquet': pl.read_parquet
    }
    df = dict_[file_type](file_path)
    return df

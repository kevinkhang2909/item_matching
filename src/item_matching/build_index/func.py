from pathlib import Path
import polars as pl


class PipelineText:
    @staticmethod
    def clean_text(data: pl.DataFrame, col: str = 'item_name') -> pl.DataFrame:
        import re

        regex = "[\(\[\<\"].*?[\)\]\>\"]"
        return data.with_columns(
            pl.col(col).map_elements(
                lambda x: re.sub(regex, "", x).lower().rstrip('.').strip(), return_dtype=pl.String
            )
            .alias(f'{col.lower()}_clean')
        )


def ngrams_func(string: str):
    ngrams = zip(*[string[i:] for i in range(3)])
    return [''.join(ngram) for ngram in ngrams]


def tfidf(lst_item: list, dim: int = 512):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer=ngrams_func, max_features=dim)
    vectorizer.fit(lst_item)
    return vectorizer


def rm_all_folder(path: Path) -> None:
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_all_folder(child)
    path.rmdir()


def make_dir(folder_name: str | Path) -> None:
    """Make a directory if it doesn't exist'"""
    if isinstance(folder_name, str):
        folder_name = Path(folder_name)
    if not folder_name.exists():
        folder_name.mkdir(parents=True, exist_ok=True)

from pathlib import Path
import polars as pl


def rm_all_folder(path: Path) -> None:
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_all_folder(child)
    path.rmdir()


def make_dir(folder_name: str | Path) -> None:
    """Make a directory if it doesn't exist"""
    if isinstance(folder_name, str):
        folder_name = Path(folder_name)
    if not folder_name.exists():
        folder_name.mkdir(parents=True, exist_ok=True)


def clean_check(data: pl.DataFrame, pct: bool = False, verbose: bool = False) -> dict:
    null_check_dict = data.null_count().to_dicts()[0].items()
    null_check = {i: v for i, v in null_check_dict if v > 0}
    if pct:
        null_check = {i: v / data.shape[0] for i, v in null_check_dict if v > 0}

    if verbose:
        print(f"- Null columns: {null_check}")
    dict_ = {
        'null': null_check,
    }
    return dict_

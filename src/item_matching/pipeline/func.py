from core_pro.ultilities import make_dir
from pathlib import Path
import numpy as np


def _create_folder(path: Path, folder_name: str, one: bool = False):
    if one:
        new_folder = path / folder_name
        make_dir(new_folder)
        return new_folder
    else:
        dict_ = {}
        for i in ["db", "q"]:
            folder = f"{i}_{folder_name}"
            new_folder = path / folder
            make_dir(new_folder)
            dict_[i] = new_folder
    return dict_


def _round_score(arr: np.array, decimals: int = 5) -> list:
    return np.where(arr <= 1, np.round(arr, decimals), np.round(arr, 1)).tolist()

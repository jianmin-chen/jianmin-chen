from os import path
from pathlib import Path
import numpy as np

from .download_utils import download

_path = Path(path.dirname(path.abspath(__file__)))

__all__ = ["load", "download", "get_path"]

def _get_dataset_path(dataset_name):
    return _path / dataset_name

def get_path():
    return _get_dataset_path("cifar-10-python.npz")

def load(fname="cifar-10-python.npz"):
    if not (_path / fname).exists():
        msg = """Date not found! Please download the data (cifar-10-python.npz) using `cifar10.download()`"""
        raise FileNotFoundError(msg)

    with np.load(str(_path / fname)) as data:
        xtr, ytr, xte, yte = tuple(data[key] for key in ["x_train", "y_train", "x_test", "y_test"])
    print("cifar-10 loaded")
    return xtr, ytr, xte, yte

load.labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

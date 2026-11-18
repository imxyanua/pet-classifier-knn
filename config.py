# code by xyanua - github.com/imxyanua
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(ROOT_DIR, "raw_data")
TAR_PATH = os.path.join(RAW_DIR, "images.tar.gz")
CACHE_PATH = os.path.join(ROOT_DIR, "features_cache.npz")

SELECTED_BREEDS = [
    "pug",
    "samoyed",
    "Bombay",
    "Egyptian_Mau",
    "Sphynx",
]

DISPLAY_NAMES = {
    "pug": "Pug",
    "samoyed": "Samoyed",
    "Bombay": "Bombay",
    "Egyptian_Mau": "Egyptian Mau",
    "Sphynx": "Sphynx",
}

K_CHOICES = [1, 3, 5, 7, 9, 11, 13, 15, 19, 25]
METRIC_LABELS = {"Euclidean": "euclidean", "Manhattan": "manhattan"}

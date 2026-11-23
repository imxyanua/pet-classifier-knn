# code by xyanua - github.com/imxyanua
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(ROOT_DIR, "raw_data")
TAR_PATH = os.path.join(RAW_DIR, "images.tar.gz")
ANNOTATIONS_TAR_PATH = os.path.join(RAW_DIR, "annotations.tar.gz")
TRIMAP_DIR = os.path.join(RAW_DIR, "annotations", "trimaps")
CACHE_PATH = os.path.join(ROOT_DIR, "features_cache.npz")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
EVAL_JSON_PATH = os.path.join(DOCS_DIR, "evaluation_summary.json")

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

# Hu moments vẫn được trích và cache, nhưng mặc định không dùng: riêng chúng gần mức đoán bừa.
DEFAULT_FEATURE_GROUPS = ["color", "geometry"]
METRIC_LABELS = {
    "Euclidean": "euclidean",
    "Manhattan": "manhattan",
    "Cosine": "cosine",
    "Chi-square": "chi_square",
}
VOTE_LABELS = {"Đều": "uniform", "Theo khoảng cách": "distance"}

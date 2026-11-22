# code by xyanua - github.com/imxyanua
import os
import numpy as np

from config import CACHE_PATH, DATA_DIR, SELECTED_BREEDS
from features import extract_features, load_bgr

CACHE_VERSION = 2


def list_samples(data_dir=DATA_DIR):
    samples = []
    for breed in SELECTED_BREEDS:
        folder = os.path.join(data_dir, breed)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(folder, name), breed))
    return samples


def load_feature_cache(path=CACHE_PATH):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    if int(data["version"]) != CACHE_VERSION:
        return None
    return data["X"], data["y"], data["paths"]


def save_feature_cache(X, y, paths, path=CACHE_PATH):
    np.savez(
        path,
        version=np.array(CACHE_VERSION),
        X=X,
        y=np.array(y),
        paths=np.array(paths),
    )


def extract_dataset(samples=None, progress=None):
    if samples is None:
        samples = list_samples()
    if not samples:
        raise FileNotFoundError(
            "Không thấy ảnh trong thư mục data/. Hãy chạy download_data.py trước."
        )

    vectors = []
    labels = []
    paths = []
    total = len(samples)
    for i, (path, label) in enumerate(samples, start=1):
        vector, _, _ = extract_features(load_bgr(path))
        vectors.append(vector)
        labels.append(label)
        paths.append(path)
        if progress:
            progress(i, total, path)
    return np.vstack(vectors), np.array(labels), np.array(paths)


def load_or_extract(progress=None, force=False):
    if not force:
        cached = load_feature_cache()
        if cached is not None:
            return cached
    X, y, paths = extract_dataset(progress=progress)
    save_feature_cache(X, y, paths)
    return X, y, paths

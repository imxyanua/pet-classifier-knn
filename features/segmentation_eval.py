# code by xyanua - github.com/imxyanua
"""Chấm điểm phân đoạn GrabCut bằng trimap của bộ Oxford-IIIT Pet.

Trimap chỉ dùng để đo chất lượng mask; đặc trưng và k-NN không bao giờ dùng trimap.
Giá trị trimap: 1 = con vật, 2 = nền, 3 = viền (bỏ qua khi tính điểm).
"""
import os
from multiprocessing import Pool

import cv2
import numpy as np

from config import DISPLAY_NAMES, SELECTED_BREEDS, TRIMAP_DIR
from dataset import list_samples

from .features import load_bgr, segment_pet

TRIMAP_FG = 1
TRIMAP_BORDER = 3
# Một số trimap trong bộ Oxford bị hỏng (gần như không có pixel con vật).
MIN_TRIMAP_FG = 0.01


def load_trimap(image_path, shape):
    name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    path = os.path.join(TRIMAP_DIR, name)
    trimap = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        raise FileNotFoundError(
            f"Không thấy trimap {path}. Chạy: python download_data.py --trimaps"
        )
    return cv2.resize(trimap, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def mask_scores(mask, trimap):
    """IoU, precision, recall của mask so với trimap, bỏ qua vùng viền."""
    valid = trimap != TRIMAP_BORDER
    truth = (trimap == TRIMAP_FG)[valid]
    pred = (mask > 0)[valid]
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    iou = inter / union if union else 1.0
    precision = inter / pred.sum() if pred.sum() else 0.0
    recall = inter / truth.sum() if truth.sum() else 1.0
    return float(iou), float(precision), float(recall)


def _score_one(sample):
    path, label = sample
    img, mask = segment_pet(load_bgr(path))
    trimap = load_trimap(path, img.shape)
    if (trimap == TRIMAP_FG).mean() < MIN_TRIMAP_FG:
        return None
    return (label, *mask_scores(mask, trimap))


def main():
    samples = list_samples()
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Đang phân đoạn và chấm điểm {len(samples)} ảnh…")
    with Pool(workers) as pool:
        results = pool.map(_score_one, samples, chunksize=8)

    skipped = sum(r is None for r in results)
    results = [r for r in results if r is not None]
    labels = np.array([r[0] for r in results])
    scores = np.array([r[1:] for r in results])
    print(f"{len(results)} ảnh (bỏ {skipped} ảnh có trimap hỏng)\n")

    print(f"{'giống':<16}{'IoU':>8}{'IoU<0.5':>10}{'precision':>11}{'recall':>9}")
    for breed in SELECTED_BREEDS + [None]:
        rows = scores if breed is None else scores[labels == breed]
        name = "tất cả" if breed is None else DISPLAY_NAMES[breed]
        print(
            f"{name:<16}{rows[:, 0].mean():8.3f}{(rows[:, 0] < 0.5).mean() * 100:9.1f}%"
            f"{rows[:, 1].mean():11.3f}{rows[:, 2].mean():9.3f}"
        )


if __name__ == "__main__":
    main()

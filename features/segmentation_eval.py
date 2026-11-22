# code by xyanua - github.com/imxyanua
"""Chấm điểm phân đoạn GrabCut bằng trimap của bộ Oxford-IIIT Pet.

Trimap chỉ dùng để đo chất lượng mask; đặc trưng và k-NN không bao giờ dùng trimap.
Giá trị trimap: 1 = con vật, 2 = nền, 3 = viền (bỏ qua khi tính điểm).
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np

from config import DISPLAY_NAMES, SELECTED_BREEDS, TRIMAP_DIR
from dataset import list_samples
from knn import stratified_split

from .features import load_bgr, segment_pet

TRIMAP_FG = 1
TRIMAP_BORDER = 3
# Một số trimap trong bộ Oxford bị hỏng (gần như không có pixel con vật).
MIN_TRIMAP_FG = 0.01
BAD_IOU = 0.5
# Dưới ngưỡng này: recall thấp = thiếu thân, precision thấp = dính nền.
LOW_SCORE = 0.6
# Trùng với seed chia train/test mặc định của evaluate.py.
SPLIT_SEED = 42


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


def failure_type(iou, precision, recall):
    """None nếu mask đạt; nếu sai nặng thì 'missing' (thiếu thân), 'background' (dính nền) hoặc 'both'."""
    if iou >= BAD_IOU:
        return None
    missing = recall < LOW_SCORE
    background = precision < LOW_SCORE
    if missing and background:
        return "both"
    if missing:
        return "missing"
    if background:
        return "background"
    return "both"


def split_samples(samples, which):
    """Lấy phần train hoặc test theo cùng cách chia với evaluate.py (seed 42)."""
    if which == "all":
        return samples
    labels = np.array([label for _, label in samples])
    train_idx, test_idx = stratified_split(labels, test_ratio=0.2, seed=SPLIT_SEED)
    idx = train_idx if which == "train" else test_idx
    return [samples[i] for i in sorted(idx)]


def _score_one(segment_fn, sample):
    path, label = sample
    img, mask = segment_fn(load_bgr(path))
    trimap = load_trimap(path, img.shape)
    if (trimap == TRIMAP_FG).mean() < MIN_TRIMAP_FG:
        return None
    return (label, *mask_scores(mask, trimap))


def score_samples(samples, segment_fn=segment_pet, workers=None):
    """Phân đoạn từng ảnh bằng segment_fn và chấm điểm.

    Trả về (labels, scores, skipped); scores có các cột IoU, precision, recall.
    segment_fn phải là hàm cấp module để chạy được trong tiến trình con.
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    with Pool(workers) as pool:
        results = pool.map(partial(_score_one, segment_fn), samples, chunksize=8)
    skipped = sum(r is None for r in results)
    results = [r for r in results if r is not None]
    labels = np.array([r[0] for r in results])
    scores = np.array([r[1:] for r in results], dtype=np.float64)
    return labels, scores, skipped


def print_report(labels, scores):
    kinds = np.array([failure_type(*row) for row in scores], dtype=object)
    header = (
        f"{'giống':<16}{'IoU':>7}{'IoU<0.5':>9}{'precision':>11}{'recall':>8}"
        f"{'thiếu thân':>12}{'dính nền':>10}{'cả hai':>8}"
    )
    print(header)
    for breed in SELECTED_BREEDS + [None]:
        sel = np.ones(len(labels), bool) if breed is None else labels == breed
        rows, k = scores[sel], kinds[sel]
        name = "tất cả" if breed is None else DISPLAY_NAMES[breed]
        print(
            f"{name:<16}{rows[:, 0].mean():7.3f}{(rows[:, 0] < BAD_IOU).mean() * 100:8.1f}%"
            f"{rows[:, 1].mean():11.3f}{rows[:, 2].mean():8.3f}"
            f"{np.sum(k == 'missing'):12d}{np.sum(k == 'background'):10d}{np.sum(k == 'both'):8d}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chấm điểm phân đoạn bằng trimap.")
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="chấm trên toàn bộ ảnh, hoặc chỉ phần train/test của evaluate.py (seed 42)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    samples = split_samples(list_samples(), args.split)
    print(f"Đang phân đoạn và chấm điểm {len(samples)} ảnh ({args.split})…")
    labels, scores, skipped = score_samples(samples)
    print(f"{len(labels)} ảnh (bỏ {skipped} ảnh có trimap hỏng)")
    print(
        f"Sai nặng = IoU < {BAD_IOU}; thiếu thân = recall < {LOW_SCORE}; "
        f"dính nền = precision < {LOW_SCORE}\n"
    )
    print_report(labels, scores)


if __name__ == "__main__":
    main()

# code by xyanua - github.com/imxyanua
"""Đánh giá k-NN: chọn (khoảng cách, vote, k) bằng cross-validation trên train, báo cáo trên test."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_FEATURE_GROUPS,
    DISPLAY_NAMES,
    DOCS_DIR,
    EVAL_JSON_PATH,
    K_CHOICES,
    SELECTED_BREEDS,
)
from dataset import load_or_extract
from features import FEATURE_GROUPS, histogram_mask, select_feature_groups
from knn import (
    accuracy_score,
    confusion_matrix,
    knn_predict_batch,
    per_class_metrics,
    select_params,
    stratified_split,
)
from knn.validation import prepare_features

METRICS = ["euclidean", "manhattan", "cosine", "chi_square"]
VOTES = ["uniform", "distance"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="seed chia train/test cho báo cáo chi tiết")
    parser.add_argument("--folds", type=int, default=5, help="số fold cross-validation trên tập train")
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="lặp toàn bộ quy trình với seed 0..N-1 để ước lượng độ dao động (0 để bỏ qua)",
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURE_GROUPS),
        help=f"nhóm đặc trưng, phân tách bằng dấu phẩy: {', '.join(FEATURE_GROUPS)}",
    )
    parser.add_argument(
        "--export-dir",
        default=DOCS_DIR,
        help="thư mục ghi JSON + accuracy_k.png + confusion_matrix.png (chuỗi rỗng để bỏ)",
    )
    return parser.parse_args(argv)


def run_once(X, y, seed, n_folds, hist_mask=None):
    """Chia train/test, chọn tham số bằng CV trên train, đánh giá đúng một lần trên test."""
    train_idx, test_idx = stratified_split(y, test_ratio=0.2, seed=seed)
    (metric, vote, k, cv_acc), grid = select_params(
        X[train_idx],
        y[train_idx],
        K_CHOICES,
        METRICS,
        n_folds=n_folds,
        seed=seed,
        votes=VOTES,
        hist_mask=hist_mask,
    )
    X_train, X_test, _, _ = prepare_features(
        X[train_idx], X[test_idx], metric, hist_mask
    )
    preds = knn_predict_batch(
        X_test,
        X_train,
        y[train_idx],
        k=k,
        metric=metric,
        vote=vote,
        hist_mask=hist_mask,
    )
    return {
        "metric": metric,
        "vote": vote,
        "k": k,
        "cv_acc": cv_acc,
        "grid": grid,
        "test_acc": accuracy_score(y[test_idx], preds),
        "cm": confusion_matrix(y[test_idx], preds, SELECTED_BREEDS),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def print_detailed(result, seed, n_folds):
    print(f"\n=== Chi tiết với seed={seed} ===")
    print(f"Train {result['n_train']} · Test {result['n_test']}")
    print(f"\nAccuracy cross-validation {n_folds} fold trên tập train:")
    for vote in VOTES:
        print(f"\nvote={vote}")
        print(f"{'k':>4}  " + "  ".join(f"{m:>10}" for m in METRICS))
        for k in K_CHOICES:
            row = "  ".join(
                f"{result['grid'][(m, vote, k)] * 100:9.2f}%" for m in METRICS
            )
            print(f"{k:>4}  {row}")

    print(
        f"\nChọn theo CV: {result['metric']}, vote={result['vote']}, k={result['k']} "
        f"(CV {result['cv_acc'] * 100:.2f}%)"
    )
    print(f"Accuracy trên test: {result['test_acc'] * 100:.2f}%")

    cm = result["cm"]
    print("\nConfusion matrix (hàng = thực tế, cột = dự đoán)")
    print(" " * 16 + " ".join(f"{c[:10]:>10}" for c in SELECTED_BREEDS))
    for i, cls in enumerate(SELECTED_BREEDS):
        print(f"{cls[:16]:<16}" + " ".join(f"{n:10d}" for n in cm[i]))

    precision, recall, f1 = per_class_metrics(cm)
    print(f"\n{'lớp':<16}{'precision':>10}{'recall':>10}{'F1':>10}")
    for i, cls in enumerate(SELECTED_BREEDS):
        print(
            f"{DISPLAY_NAMES[cls]:<16}{precision[i] * 100:9.1f}%"
            f"{recall[i] * 100:9.1f}%{f1[i] * 100:9.1f}%"
        )
    print(
        f"{'trung bình':<16}{precision.mean() * 100:9.1f}%"
        f"{recall.mean() * 100:9.1f}%{f1.mean() * 100:9.1f}%"
    )


def print_repeated(X, y, repeats, n_folds, hist_mask=None):
    print(f"\n=== Lặp {repeats} lần (seed 0..{repeats - 1}) ===")
    accs = []
    chosen = Counter()
    seed_rows = []
    for seed in range(repeats):
        result = run_once(X, y, seed, n_folds, hist_mask=hist_mask)
        accs.append(result["test_acc"])
        chosen[(result["metric"], result["vote"], result["k"])] += 1
        seed_rows.append(
            {
                "seed": seed,
                "metric": result["metric"],
                "vote": result["vote"],
                "k": result["k"],
                "cv_acc": result["cv_acc"],
                "test_acc": result["test_acc"],
            }
        )
        print(
            f"seed {seed:>2}: chọn {result['metric']:<10} vote={result['vote']:<8} "
            f"k={result['k']:<2}  CV {result['cv_acc'] * 100:5.1f}%  "
            f"test {result['test_acc'] * 100:5.1f}%"
        )
    accs = np.array(accs)
    summary = {
        "mean": float(accs.mean()),
        "std": float(accs.std()),
        "min": float(accs.min()),
        "max": float(accs.max()),
        "chosen": [
            {"metric": m, "vote": v, "k": k, "count": n}
            for (m, v, k), n in chosen.most_common()
        ],
        "seeds": seed_rows,
    }
    print(
        f"\nAccuracy trên test: {summary['mean'] * 100:.1f}% ± {summary['std'] * 100:.1f}% "
        f"(thấp nhất {summary['min'] * 100:.1f}%, cao nhất {summary['max'] * 100:.1f}%)"
    )
    print(
        "Cấu hình được chọn: "
        + ", ".join(
            f"{m}/{v} k={k} ×{n}" for (m, v, k), n in chosen.most_common()
        )
    )
    return summary


def export_artifacts(result, repeats_summary, groups, export_dir, seed, n_dims):
    """Ghi JSON + vẽ lại accuracy_k.png và confusion_matrix.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    vote = result["vote"]
    grid_rows = []
    for v in VOTES:
        for metric in METRICS:
            for k in K_CHOICES:
                grid_rows.append(
                    {
                        "metric": metric,
                        "vote": v,
                        "k": k,
                        "cv_acc": float(result["grid"][(metric, v, k)]),
                    }
                )

    payload = {
        "features": groups,
        "n_dims": int(n_dims),
        "detailed_seed": {
            "seed": int(seed),
            "metric": result["metric"],
            "vote": result["vote"],
            "k": result["k"],
            "cv_acc": float(result["cv_acc"]),
            "test_acc": float(result["test_acc"]),
            "n_train": int(result["n_train"]),
            "n_test": int(result["n_test"]),
            "confusion_matrix": result["cm"].tolist(),
            "classes": list(SELECTED_BREEDS),
        },
        "cv_grid": grid_rows,
        "repeats": repeats_summary,
    }
    json_path = (
        Path(EVAL_JSON_PATH)
        if out.resolve() == Path(DOCS_DIR).resolve()
        else out / "evaluation_summary.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {
        "euclidean": "-o",
        "manhattan": "-s",
        "cosine": "-^",
        "chi_square": "-D",
    }
    for metric in METRICS:
        ys = [result["grid"][(metric, vote, k)] * 100 for k in K_CHOICES]
        ax.plot(K_CHOICES, ys, styles[metric], label=metric)
    ax.set_xlabel("k")
    ax.set_ylabel("CV accuracy (%)")
    ax.set_title(f"Cross-validation accuracy vs k (vote={vote})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_k.png", dpi=120)
    plt.close(fig)

    cm = result["cm"]
    labels = [DISPLAY_NAMES[c] for c in SELECTED_BREEDS]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Confusion matrix ({result['metric']}/{result['vote']}, k={result['k']})"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"\nĐã xuất {json_path}")
    print(f"Đã xuất {out / 'accuracy_k.png'}")
    print(f"Đã xuất {out / 'confusion_matrix.png'}")
    return json_path


def main(argv=None):
    args = parse_args(argv)
    groups = [g.strip() for g in args.features.split(",") if g.strip()]

    print("Đang nạp / trích đặc trưng…")
    X, y, _paths = load_or_extract(
        progress=lambda i, n, p: print(f"\r{i}/{n}", end="", flush=True)
    )
    try:
        hist = histogram_mask(groups)
        X = select_feature_groups(X, groups)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"{len(y)} mẫu · {X.shape[1]} chiều ({', '.join(groups)})")

    detailed = run_once(X, y, args.seed, args.folds, hist_mask=hist)
    print_detailed(detailed, args.seed, args.folds)

    repeats_summary = None
    if args.repeats > 0:
        repeats_summary = print_repeated(X, y, args.repeats, args.folds, hist_mask=hist)

    if args.export_dir:
        export_artifacts(
            detailed,
            repeats_summary,
            groups,
            args.export_dir,
            seed=args.seed,
            n_dims=X.shape[1],
        )


if __name__ == "__main__":
    main()

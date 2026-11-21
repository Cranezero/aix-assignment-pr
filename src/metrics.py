"""평가 지표, 예측 저장, 혼동행렬 그림.

모든 방법의 테스트 예측은 같은 형식으로 저장한다.
results/{method}_test_pred.csv: id, country, pred, top5, p_<CODE>...
results/{method}_metrics.json: top1, top5, macro, per_country, (학습 시간)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .config import CODES, RESULTS_DIR


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, n_classes: int = None) -> np.ndarray:
    n_classes = n_classes or len(CODES)
    return np.array([(preds[labels == c] == c).mean() if (labels == c).any() else np.nan
                     for c in range(n_classes)])


def evaluate(probs: np.ndarray, labels: np.ndarray) -> dict:
    probs, labels = np.asarray(probs), np.asarray(labels)
    preds = probs.argmax(axis=1)
    per_class = per_class_accuracy(preds, labels)
    return {
        "top1": topk_accuracy(probs, labels, 1),
        "top5": topk_accuracy(probs, labels, 5),
        "macro": float(np.nanmean(per_class)),
        "per_country": {c: float(a) for c, a in zip(CODES, per_class)},
    }


def save_results(method: str, ids, probs: np.ndarray, labels: np.ndarray,
                 extra: dict = None, results_dir: Path = RESULTS_DIR) -> dict:
    """예측 CSV와 지표 JSON을 저장하고 지표를 돌려준다."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    probs, labels = np.asarray(probs), np.asarray(labels)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    df = pd.DataFrame({
        "id": list(ids),
        "country": [CODES[i] for i in labels],
        "pred": [CODES[i] for i in probs.argmax(axis=1)],
        "top5": [" ".join(CODES[j] for j in row) for row in top5],
    })
    prob_df = pd.DataFrame(probs.round(5), columns=[f"p_{c}" for c in CODES])
    pd.concat([df, prob_df], axis=1).to_csv(results_dir / f"{method}_test_pred.csv", index=False)

    metrics = evaluate(probs, labels)
    if extra:
        metrics.update(extra)
    with open(results_dir / f"{method}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def load_pred(method: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(results_dir) / f"{method}_test_pred.csv", dtype={"id": str})


def plot_confusion(y_true, y_pred, ax=None, title: str = "", normalize: bool = True):
    """행 정규화 혼동행렬. y_true/y_pred는 국가 코드 문자열."""
    cm = confusion_matrix(y_true, y_pred, labels=CODES, normalize="true" if normalize else None)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, ax=ax, cmap="Blues", vmin=0, vmax=1 if normalize else None,
                xticklabels=CODES, yticklabels=CODES, square=True, cbar=False,
                annot=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return cm


def top_confusions(y_true, y_pred, n: int = 10) -> pd.DataFrame:
    """가장 많이 틀린 (정답, 예측) 쌍. rate는 정답 국가 테스트 장수 대비 비율."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    total = df["true"].value_counts()
    wrong = df[df["true"] != df["pred"]].value_counts().rename("count").reset_index()
    wrong["rate"] = wrong["count"] / wrong["true"].map(total)
    return wrong.head(n)

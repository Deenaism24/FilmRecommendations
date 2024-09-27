import numpy as np
import pandas as pd
from typing import Dict


def ndcg_metric(gt_items: np.ndarray, predicted: np.ndarray) -> float:
    at = len(predicted)
    relevance = np.array([1 if x in predicted else 0 for x in gt_items])
    rank_dcg = dcg(relevance)
    if rank_dcg == 0.0:
        return 0.0

    ideal_dcg = dcg(np.sort(relevance)[::-1][:at])

    if ideal_dcg == 0.0:
        return 0.0

    return rank_dcg / ideal_dcg


def dcg(scores: np.ndarray) -> float:
    return np.sum(np.divide(np.power(2, scores) - 1,
                            np.log2(np.arange(scores.shape[0], dtype=np.float64) + 2)), dtype=np.float64)


def recall_metric(gt_items: np.ndarray, predicted: np.ndarray) -> float:
    n_gt = len(gt_items)
    intersection = len(set(gt_items).intersection(set(predicted)))
    return intersection / n_gt


def evaluate_recommender(df: pd.DataFrame, model_preds: str, gt: str = "movie_id") -> Dict[str, float]:
    metric_values = []

    for _, row in df.iterrows():
        metric_values.append((ndcg_metric(row[gt], row[model_preds]), recall_metric(row[gt], row[model_preds])))

    return {"ndcg": np.mean([x[0] for x in metric_values]), "recall": np.mean([x[1] for x in metric_values])}

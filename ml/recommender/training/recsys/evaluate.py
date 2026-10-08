"""Evaluation harness shared by every phase.

A model is anything with:
    item_ids: np.ndarray                 movie ids the model can recommend
    score(user_ids) -> np.ndarray        shape (len(user_ids), len(item_ids)), higher = better
and optionally, for rating prediction:
    predict(user_ids, movie_ids) -> np.ndarray
"""

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate and evaluate RMSE (root-mean-square error) metric
    of a prediction
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def top_k(model, user_ids: np.ndarray, history: pd.DataFrame, k: int) -> np.ndarray:
    """Top-k movie ids per user, skipping movies the user already rated in `history`."""
    scores = np.array(model.score(user_ids), dtype=float)  # copy: we overwrite seen items
    rows = pd.Index(user_ids).get_indexer(history["userId"])
    cols = pd.Index(model.item_ids).get_indexer(history["movieId"])
    keep = (rows >= 0) & (cols >= 0)
    scores[rows[keep], cols[keep]] = -np.inf

    # argpartition finds the k best in O(n); only those k then need sorting.
    best = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, best, axis=1), axis=1)
    return model.item_ids[np.take_along_axis(best, order, axis=1)]


def evaluate(
    model, history: pd.DataFrame, test: pd.DataFrame, k: int = 10, threshold: float = 4.0
) -> dict[str, float]:
    """Score `model` on `test`, treating `history` as what each user had already rated.

    Ranking metrics count a test movie as relevant if the user rated it >= threshold.
    Users with no relevant test movies are skipped for ranking.
    """
    results: dict[str, float] = {}
    if hasattr(model, "predict"):
        pred = model.predict(test["userId"].to_numpy(), test["movieId"].to_numpy())
        results["rmse"] = rmse(test["rating"].to_numpy(), pred)

    relevant = test[test["rating"] >= threshold].groupby("userId")["movieId"].agg(set)
    recs = top_k(model, relevant.index.to_numpy(), history, k)
    hits = np.array([[m in rel for m in row] for row, rel in zip(recs, relevant)], dtype=float)
    n_rel = relevant.map(len).to_numpy()

    # NDCG: a hit at rank r is worth 1/log2(r+1); divide by the best achievable score.
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discount # reminder: @ is matrix multiplication operator (equivalent to np.matmul(a, b))
    idcg = np.cumsum(discount)[np.minimum(n_rel, k) - 1]

    results[f"recall@{k}"] = float(np.mean(hits.sum(axis=1) / n_rel))
    results[f"ndcg@{k}"] = float(np.mean(dcg / idcg))
    # Share of the catalog that shows up in anyone's top-k. Low = everyone sees the same movies.
    results[f"coverage@{k}"] = len(np.unique(recs)) / len(model.item_ids)
    results["users"] = len(relevant)
    return results

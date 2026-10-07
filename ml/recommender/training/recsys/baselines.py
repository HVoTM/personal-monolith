"""Phase 1 baselines. Every later model has to beat these."""

import numpy as np
import pandas as pd


class Popularity:
    """Recommend the most-rated movies to everyone. No rating prediction.
    
    Score based on how much rating it has. Everyone will get the same list
    """

    def fit(self, train: pd.DataFrame) -> "Popularity":
        counts = train.groupby("movieId").size()
        self.item_ids = counts.index.to_numpy()
        self._counts = counts.to_numpy(dtype=float)
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return np.tile(self._counts, (len(user_ids), 1))


class BiasModel:
    """rating ≈ mu + b_user + b_movie, with L2 regularization on the biases.

    Minimizes  sum (r - mu - b_u - b_i)^2  +  reg_user * sum b_u^2  +  reg_item * sum b_i^2.
    Holding b_u fixed, the best b_i has a closed form (and vice versa), so we alternate between
    the two. Each b_i is then a "shrunk" average residual: sum of residuals / (n_ratings + reg).
    A movie with 2 ratings gets pulled hard toward 0; one with 300 barely moves.
    """

    def __init__(self, reg_user: float = 10.0, reg_item: float = 10.0, n_iters: int = 10):
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.n_iters = n_iters

    def fit(self, train: pd.DataFrame) -> "BiasModel":
        u, users = pd.factorize(train["userId"])
        i, items = pd.factorize(train["movieId"])
        r = train["rating"].to_numpy(dtype=float)

        self.mu = r.mean()
        n_u = np.bincount(u)
        n_i = np.bincount(i)
        b_u = np.zeros(len(users))
        b_i = np.zeros(len(items))
        for _ in range(self.n_iters):
            b_i = np.bincount(i, weights=r - self.mu - b_u[u]) / (n_i + self.reg_item)
            b_u = np.bincount(u, weights=r - self.mu - b_i[i]) / (n_u + self.reg_user)

        self._users = pd.Index(users)
        self._items = pd.Index(items)
        self.item_ids = items.to_numpy()
        self.b_user = b_u
        self.b_item = b_i
        return self

    def _user_bias(self, user_ids) -> np.ndarray:
        idx = self._users.get_indexer(user_ids)
        return np.where(idx >= 0, self.b_user[idx], 0.0)  # unseen user: bias 0

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        idx = self._items.get_indexer(movie_ids)
        b_i = np.where(idx >= 0, self.b_item[idx], 0.0)
        return np.clip(self.mu + self._user_bias(user_ids) + b_i, 0.5, 5.0)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        # b_user shifts every movie equally for a given user, so it never changes the order:
        # for ranking, this model shows everyone the same list, sorted by b_item.
        return self.mu + self._user_bias(user_ids)[:, None] + self.b_item[None, :]

"""Phase 2: content-based filtering.

Each movie becomes a TF-IDF vector over three kinds of tokens:
    genre:Comedy        from movies.csv
    tag:dark comedy     from tags.csv (free text, lowercased)
    decade:1990s        parsed from the title's "(1995)"
Two movies are similar when their vectors point the same way (cosine similarity).
A user's taste is a weighted sum of the vectors of movies they rated.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def movie_tokens(movies: pd.DataFrame, tags: pd.DataFrame, min_tag_movies: int = 2) -> pd.DataFrame:
    """Long table of (movieId, token, field, count): how often each token applies to each movie."""
    genres = movies[["movieId"]].assign(token=movies["genres"].str.split("|")).explode("token")
    genres = genres[genres["token"] != "(no genres listed)"]
    genre_rows = genres.assign(token="genre:" + genres["token"], field="genre", count=1)

    # A tag's count is how many times users applied it to the movie. Tags that appear on only
    # one movie can't make two movies similar; they'd only dilute that movie's vector.
    tag_text = tags["tag"].str.lower().str.strip()
    tag_rows = tags.assign(token="tag:" + tag_text).groupby(["movieId", "token"]).size().rename("count").reset_index()
    n_movies_per_tag = tag_rows.groupby("token")["movieId"].transform("size")
    tag_rows = tag_rows[n_movies_per_tag >= min_tag_movies].assign(field="tag")

    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("float")
    has_year = year.notna()
    decade = (year[has_year] // 10 * 10).astype(int).astype(str)
    decade_rows = pd.DataFrame(
        {"movieId": movies.loc[has_year, "movieId"], "token": "decade:" + decade + "s", "field": "decade", "count": 1}
    )

    return pd.concat([genre_rows, tag_rows, decade_rows], ignore_index=True)[["movieId", "token", "field", "count"]]


def tfidf(
    tokens: pd.DataFrame, item_ids: np.ndarray, field_weights: dict[str, float]
) -> tuple[sp.csr_matrix, pd.Index]:
    """Sparse (n_items × vocab) matrix with L2-normalized rows, plus the vocabulary.

    weight = tf × idf × field_weight, where
        tf  = 1 + log(count)                 a tag applied 10 times isn't 10× as informative as once
        idf = log((1 + N) / (1 + df)) + 1     rare tokens count more; "genre:Drama" (on ~45% of
                                              movies) says less about a movie than "tag:time travel"
    Rows are then scaled to length 1, so a dot product between two rows is their cosine similarity.
    """
    tokens = tokens[tokens["field"].map(field_weights).fillna(0) > 0]
    rows = pd.Index(item_ids).get_indexer(tokens["movieId"])
    cols, vocab = pd.factorize(tokens["token"])

    n_items = len(item_ids)
    df = np.bincount(cols, minlength=len(vocab))
    idf = np.log((1 + n_items) / (1 + df)) + 1
    tf = 1 + np.log(tokens["count"].to_numpy(dtype=float))
    w = tf * idf[cols] * tokens["field"].map(field_weights).to_numpy(dtype=float)

    X = sp.csr_matrix((w, (rows, cols)), shape=(n_items, len(vocab)))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sp.diags(1 / np.where(norms > 0, norms, 1)) @ X  # movies with no tokens stay all-zero
    return X.tocsr(), pd.Index(vocab)


class ContentModel:
    """Recommend movies whose content resembles what the user liked.

    profile:
        "centered"  weight each rated movie by (rating - user's mean): disliked movies push away
        "liked"     weight 1 for ratings >= 4, ignore the rest
        "rating"    weight by the raw rating: everything rated pulls toward itself

    popularity_weight (β):
        0   pure content: score = profile · movie
        > 0 score = content / max|content| + β × log(1 + n_ratings in train)
            Content says what a movie is about, not whether it's any good; popularity fills that
            gap. Scaling content to [-1, 1] per user makes β mean the same thing for every user.
            A user with no ratings gets pure popularity.
    """

    def __init__(
        self,
        genre_weight: float = 1.0,
        tag_weight: float = 1.0,
        decade_weight: float = 0.5,
        min_tag_movies: int = 2,
        profile: str = "centered",
        popularity_weight: float = 0.0,
    ):
        self.field_weights = {"genre": genre_weight, "tag": tag_weight, "decade": decade_weight}
        self.min_tag_movies = min_tag_movies
        self.profile = profile
        self.popularity_weight = popularity_weight

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame, tags: pd.DataFrame) -> "ContentModel":
        # The whole catalog is recommendable, including movies nobody has rated yet.
        self.item_ids = movies["movieId"].to_numpy()
        self._items = pd.Index(self.item_ids)
        tokens = movie_tokens(movies, tags, self.min_tag_movies)
        self.X, self.vocab = tfidf(tokens, self.item_ids, self.field_weights)

        u, users = pd.factorize(train["userId"])
        r = train["rating"].to_numpy(dtype=float)
        if self.profile == "centered":
            w = r - (np.bincount(u, weights=r) / np.bincount(u))[u]
        elif self.profile == "liked":
            w = (r >= 4).astype(float)
        elif self.profile == "rating":
            w = r
        else:
            raise ValueError(f"unknown profile {self.profile!r}")

        # W is (users × items) with the weights; W @ X sums each user's weighted movie vectors.
        W = sp.csr_matrix((w, (u, self._items.get_indexer(train["movieId"]))), shape=(len(users), len(self.item_ids)))
        self.profiles = (W @ self.X).tocsr()
        self._users = pd.Index(users)

        counts = train.groupby("movieId").size().reindex(self.item_ids, fill_value=0)
        self._log_pop = np.log1p(counts.to_numpy(dtype=float))
        return self

    def _profile_rows(self, user_ids) -> sp.csr_matrix:
        idx = self._users.get_indexer(user_ids)
        known = sp.diags((idx >= 0).astype(float))  # unseen user: all-zero profile
        return (known @ self.profiles[np.where(idx >= 0, idx, 0)]).tocsr()

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        # Dot product with each movie. Not divided by the profile's length: that would scale a
        # user's whole row by one number, which doesn't change their ranking.
        s = (self._profile_rows(user_ids) @ self.X.T).toarray()
        if self.popularity_weight == 0:
            return s
        peak = np.abs(s).max(axis=1, keepdims=True)
        s = s / np.where(peak > 0, peak, 1)
        return s + self.popularity_weight * self._log_pop[None, :]

    def similar(self, movie_id: int, k: int = 10) -> pd.Series:
        """The k movies with the highest cosine similarity to movie_id, as a movieId → similarity Series."""
        i = self._items.get_loc(movie_id)
        sims = (self.X @ self.X[i].T).toarray().ravel()
        sims[i] = -np.inf
        best = np.argsort(-sims)[:k]
        return pd.Series(sims[best], index=pd.Index(self.item_ids[best], name="movieId"), name="similarity")

    def explain(self, movie_a: int, movie_b: int, n: int = 5) -> pd.Series:
        """Tokens contributing most to the similarity of two movies (the terms of their dot product)."""
        a = self.X[self._items.get_loc(movie_a)]
        b = self.X[self._items.get_loc(movie_b)]
        contrib = a.multiply(b).toarray().ravel()
        top = np.argsort(-contrib)[:n]
        top = top[contrib[top] > 0]
        return pd.Series(contrib[top], index=self.vocab[top], name="contribution")

    def movie_vector(self, movie_id: int, n: int = 10) -> pd.Series:
        """The movie's highest-weighted tokens."""
        row = self.X[self._items.get_loc(movie_id)].toarray().ravel()
        top = np.argsort(-row)[:n]
        top = top[row[top] > 0]
        return pd.Series(row[top], index=self.vocab[top], name="weight")

    def taste(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """The user's profile in words: tokens they lean toward (+) and away from (-)."""
        row = self._profile_rows([user_id]).toarray().ravel()
        order = np.argsort(-row)
        likes, dislikes = order[:n], order[::-1][:n]
        return pd.DataFrame(
            {
                "leans toward": self.vocab[likes],
                "+": row[likes].round(2),
                "leans away from": self.vocab[dislikes],
                "-": row[dislikes].round(2),
            }
        )

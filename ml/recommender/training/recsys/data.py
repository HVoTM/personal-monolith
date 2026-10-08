"""Loading MovieLens and splitting ratings into train / val / test."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ml-latest-small"


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Columns: userId, movieId, rating (0.5-5.0), timestamp (unix seconds)."""
    path = data_dir / "ratings.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run training/download_data.py first")
    return pd.read_csv(path)


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Columns: movieId, title, genres (pipe-separated)."""
    return pd.read_csv(data_dir / "movies.csv")


def load_tags(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Columns: userId, movieId, tag (free text), timestamp."""
    return pd.read_csv(data_dir / "tags.csv")


def drop_pairs(df: pd.DataFrame, *heldout: pd.DataFrame) -> pd.DataFrame:
    """Rows of df whose (userId, movieId) pair isn't in any heldout frame.

    Used on tags: a user who tagged a movie has seen it. Keeping tags for held-out ratings
    would let movie features carry a trace of the answers we're testing against.
    """
    pairs = pd.concat([h[["userId", "movieId"]] for h in heldout]).drop_duplicates()
    merged = df.merge(pairs, on=["userId", "movieId"], how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)


def time_split(
    ratings: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split: each user's newest ratings go to test, the ones before to val.

    A global cutoff date would be stricter, but most MovieLens users rate everything in one
    sitting, so almost every test user would be missing from train (see 00_explore section 6).
    The trade-off: train can contain other users' ratings made after a test rating.
    """
    df = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    size = df.groupby("userId")["movieId"].transform("size").to_numpy()
    from_end = size - 1 - df.groupby("userId").cumcount().to_numpy()  # 0 = user's newest rating

    n_test = np.ceil(size * test_frac)
    n_val = np.ceil(size * val_frac)
    is_test = from_end < n_test
    is_val = ~is_test & (from_end < n_test + n_val)

    train = df[~is_test & ~is_val].reset_index(drop=True)
    val = df[is_val].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, val, test

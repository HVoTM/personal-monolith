# recommender

A movie recommender that learns a user's taste from their ratings and serves recommendations through a small API. It's a learning project for recommender systems.

- Training: Python + TensorFlow/Keras
- Serving: Go (to match the rest of the monolith) or FastAPI
- Data: [MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/) (100k ratings, 9k movies)

See [CONCEPTS.md](CONCEPTS.md) for the background theory.

## Setup

Requires Python 3.12 (TensorFlow, added in phase 3, doesn't support every new Python release right away).

```powershell
cd apps/recommender
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r training/requirements.txt
python training/download_data.py      # → data/raw/ml-latest-small/
jupyter lab training/notebooks/            # 00_explore, 01_baselines, ...
```

## Roadmap

- [ ] **Phase 0: Data.** Download MovieLens-small and explore it in a notebook: rating distribution, long tail, sparsity.
- [ ] **Phase 1: Baselines.** Build a popularity recommender, then the bias model (`mean + user bias + movie bias`). Write the evaluation harness now (time split, RMSE, Recall@10, NDCG@10) and reuse it in every later phase.
- [ ] **Phase 2: Content-based.** TF-IDF on genres and tags, cosine similarity, a "similar movies" function.
- [ ] **Phase 3: Matrix factorization in TF.** Keras model: `Embedding(user) · Embedding(movie)` + biases, trained with MSE + L2. Tune the embedding dimension (16–64).
- [ ] **Phase 4: Two-tower retrieval.** Same idea, trained for ranking with a softmax loss and in-batch negatives. Add genre and year features to the movie tower. Compare Recall@10 against phase 3.
- [ ] **Phase 5: Export.** Save the movie embedding matrix and the id map (`.npy` or JSON).
- [ ] **Phase 6: API.** Implement the handlers below. Use fold-in so new ratings change recommendations immediately.
- [ ] **Phase 7: Extras.** Diversity and novelty re-ranking, filtering out already-seen movies, periodic retraining, ANN search (FAISS/ScaNN) if the catalog gets big.

**Rule of thumb:** phases 1–3 matter most. If the factorization model can't beat the bias baseline on NDCG, look for a bug before adding complexity.

## API

```
GET  /movies?search=inception          → find movie ids
POST /users                            → create user
POST /users/{id}/ratings               → {movieId, rating}; triggers fold-in
GET  /users/{id}/recommendations?k=10  → top-K by dot product, minus already-rated
GET  /movies/{id}/similar?k=10         → nearest neighbours in embedding space
```

## Layout

```
recommender/
  training/   # Python: notebooks, TF models, eval, export script
    recsys/   #   shared code: data loading + split, evaluation harness, models
  model/      # exported embeddings + id maps (gitignored)
  server/     # API
```

At serving time the trained model is just a matrix of ~9k × 32 floats. Brute-force dot products over 9k movies take microseconds, so the server needs no TensorFlow at runtime. It loads the exported embeddings and does fold-in plus top-K itself.

## Notes

- **Windows:** native Windows TensorFlow has been CPU-only since 2.10. CPU is fine for MovieLens-small; use WSL2 if you want a GPU later.
- **TensorFlow Recommenders (TFRS)** has two-tower helpers but is tied to older Keras (`tf_keras`). Write phases 3–4 in plain Keras first so you understand each part, then compare with TFRS.

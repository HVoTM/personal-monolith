# Concepts

Learn these roughly in order. Each one maps to a phase in the [roadmap](README.md#roadmap).

**Prerequisites:** vectors, dot products and cosine similarity; gradient descent and loss functions; Python with numpy and pandas.

## 1. Framing the problem

- **Explicit vs implicit feedback.** Explicit feedback is a star rating. Implicit feedback is a signal like "watched it" or "clicked it". The math and loss functions differ.
- **Rating prediction vs ranking.** Rating prediction asks "what would this user rate movie X?". Ranking asks "which 10 movies should we show?". Real systems care about ranking.

## 2. Baselines

- **Most popular:** recommend the same top movies to everyone.
- **Bias model:** `global mean + user bias + movie bias`.
- These beat fancier models more often than you'd expect. Every later model has to beat them.

## 3. Content-based filtering

- Represent movies by genres, tags and plot keywords as **TF-IDF** vectors.
- Use **cosine similarity** to answer "more like this".
- Needs no data from other users.

## 4. Collaborative filtering

- The user × movie matrix is very sparse (~99% empty).
- **Item-item neighborhoods:** "people who liked X also liked Y".

## 5. Matrix factorization and embeddings

This is the core idea.

- Each user and each movie gets a learned vector (an **embedding**). Their **dot product** is the predicted affinity.
- Topics: latent factors, loss function, L2 regularization, SGD vs ALS.

## 6. Evaluation

- **Splits:** use a time-based train/val/test split so you never train on the future (leakage).
- **Rating metrics:** RMSE.
- **Ranking metrics:** Precision@K, Recall@K, NDCG@K.
- Watch for overfitting.

## 7. Two-tower retrieval models

- A user tower and a movie tower each output an embedding.
- Trained for ranking with a softmax loss and **in-batch negative sampling**.
- Roughly how YouTube- or Netflix-style candidate retrieval works.

## 8. Cold start and fold-in

- A brand-new user has no embedding yet.
- **Fold-in:** compute one on the fly from the movies they've rated. Either average those movies' embeddings, or solve a small least-squares problem with the movie vectors held fixed.
- Without this, the system only learns a user's taste when you retrain.

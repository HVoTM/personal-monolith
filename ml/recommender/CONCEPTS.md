# Problem

- Explicit vs implicit feedback.
    - Explicit: star rating
    - Implicit: a signal like "watched it" or "clicked it"
- Rating prediction vs ranking. Rating prediction asks "what would this user rate movie X?". Ranking asks "which 10 movies should we show?". Real systems care about ranking.

- Baselines:
    - Start with "most popular"
    - Then "global mean + user bias + movie bias"

- Content-based filtering: represent movies by genres, tags, and plot keywords (TF-IDF vectors), then use cosine similarity  to answer "more like this"
- Collaborative filtering: Look at the user × movie matrix and how sparse it is (around 99% empty). Then learn item-item neighborhoods: "people who liked X also liked Y".
- Matrix factorization and embeddings This is the core idea. Each user and each movie gets a learned vector, and their dot product is the predicted affinity. You'll need to understand latent factors, the loss function, L2 regularization, and SGD vs ALS. TensorFlow becomes useful here.
- Evaluation
   - Splits: use a time-based train/val/test split so you don't train on the future.
   - Metrics: RMSE for rating prediction; Precision@K, Recall@K and NDCG for ranking.
   - Overfitting.
- Two-tower retrieval models. A user tower and a movie tower each output an embedding, trained with in-batch negative sampling. This is roughly how YouTube- or Netflix-style candidate retrieval works.
- Cold start and fold-in. A brand-new user has no embedding yet. You compute one on the fly from the movies they've rated, using an average of those movies' embeddings or a small least-squares solve with the movie vectors held fixed. Without this, the model only learns a user's taste when you retrain.

## Bias model and regularization

Code: `training/recsys/baselines.py` (`BiasModel`).

### Model
```
r̂_ui = μ + b_u + b_i
```
- `μ`: global mean rating in train
- `b_u`: how far user u's ratings sit from average (harsh critic → negative)
- `b_i`: how far movie i's ratings sit from average, after accounting for who rated it
- No personal taste yet ("likes sci-fi"); that's the `p_u · q_i` term added in matrix factorization.

### Loss
```
L = Σ (r_ui − μ − b_u − b_i)²  +  λ_u Σ b_u²  +  λ_i Σ b_i²
    ─────── fit the data ──────    ─── L2: pull biases toward 0 ───
```
`λ_u`, `λ_i` = `reg_user`, `reg_item`. A bias only moves away from 0 when there's enough evidence for it.

### Closed-form update
With `e_ui = r_ui − μ − b_u` (the residual), set `∂L/∂b_i = 0`:
```
b_i = Σ e_ui / (n_i + λ_i)  =  n_i / (n_i + λ)  ×  mean residual
                               ─── shrink factor ───
```
- With λ = 10: 1 rating keeps 9% of the effect, 10 ratings keep 50%, 1000 keep 99%.
- λ = the number of ratings at which data and prior are trusted equally.
- Imaginary ratings: `μ + b_i = (n_i·avg_i + λ·μ) / (n_i + λ)` (when b_u = 0), like starting every movie with λ ratings at the global mean.

Example (μ = 3.5, λ = 10):

| | 2 ratings of 5.0 | 300 ratings, avg 4.3 |
|---|---|---|
| λ = 0 | **+1.50** (ranks first) | +0.80 |
| λ = 10 | +0.25 | **+0.77** (ranks first) |

### Why it works
- **Bias-variance:** `expected error = bias² + variance`. A raw average from 2 ratings is unbiased but has high variance. Shrinking adds a little bias and removes a lot of variance.
- **Bayesian view:** prior `b_i ~ N(0, τ²)`, rating noise `ε ~ N(0, σ²)`. The MAP estimate is the formula above with `λ = σ² / τ²`, the ratio of noise to how much movies really differ. L2 regularization is the same as a Gaussian prior at 0.

### Training: alternating updates
`b_i` depends on `b_u` and vice versa: a movie rated only by harsh critics looks worse than it is. So alternate: solve all `b_i` with `b_u` fixed, then all `b_u` with `b_i` fixed, and repeat. This is coordinate descent on a convex quadratic, so it converges to the global minimum. `μ` isn't regularized because it's estimated from ~100k ratings.

### Extremes

| λ | Behavior | RMSE | Ranking |
|---|---|---|---|
| 0 | Raw averages, overfits movies with few ratings | Worse on val | Obscure movies with a few perfect ratings on top |
| Tuned | Balanced | Best on val | Reasonable |
| → ∞ | All biases → 0, prediction = μ | Same as global mean | By `Σ e_ui`: "popular and liked", close to the popularity list |

- `b_u` never changes a single user's ranking, because it adds the same amount to every movie. It matters for RMSE only.

### Analogies
- **Restaurant reviews:** two 5-star reviews vs 300 reviews averaging 4.3. You start the new place at "average" and let each review move your opinion a little.
- **Batting average:** going 2-for-2 on opening day gives 1.000, but nobody expects that to last. Shrinking early averages toward the league average predicts better (Efron & Morris).
- **Harsh and lenient graders:** to judge essays fairly you need each grader's harshness, and to measure harshness you need the essays' quality. Estimate one, then the other, and repeat until neither changes. Graders = users, essays = movies.

## Evaluation
- [Root Mean Square Error (RMSE)](https://c3.ai/resources/glossary/data-science/root-mean-square-error-rmse) 
    - to measure how far the predictions fall from the measured true values using Euclidean distance

- Recall and Precision at K for recommender system: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54

- NCDG metric: https://www.evidentlyai.com/ranking-metrics/ndcg-metric
    - relevance score, k parameter, cumulative gain, discounted gain, normalized DCG
    - intepretation: NDCG vs MAP, NDCG vs MRR


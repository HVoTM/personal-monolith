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

## Content-based filtering

Code: `training/recsys/content.py` (`ContentModel`), notebook `02_content.ipynb`.

Idea: describe each movie by what it *is*, describe each user by the kind of movies they rate highly, and recommend movies that match. No other user's ratings are needed to describe a movie.

### Movies as vectors
Each movie becomes a vector with one dimension per **token**:

| Field | Example token | Source |
|---|---|---|
| genre | `genre:Animation` | `movies.csv` (pipe-separated) |
| tag | `tag:pixar` | `tags.csv`, lowercased; only tags used on ≥ 2 movies (`min_tag_movies`) |
| decade | `decade:1990s` | parsed from the title's `(1995)` |

On MovieLens-small: 9,742 movies × 475 tokens (19 genres, 12 decades, 444 tags), 0.73% filled. Only ~13% of movies have any tag. The rest are described by genres and a decade only.

### TF-IDF weighting
The name comes from text search, where "document" = movie and "term" = token.
```
weight(movie, token) = tf × idf × field_weight
    tf  = 1 + log(count)                   count = times users applied the tag (1 for genre/decade)
    idf = log((1 + N) / (1 + df)) + 1      N = movies in catalog, df = movies having the token
```
- **tf (term frequency):** a tag applied 10 times counts more than once, but not 10× more (the log flattens it).
- **idf (inverse document frequency):** common tokens say little about any one movie.
  - `genre:Drama` is on 4,361 of 9,742 movies: idf = log(9743/4362) + 1 ≈ **1.80**
  - a tag on 10 movies: idf = log(9743/11) + 1 ≈ **7.79**, about 4× the weight
- **field_weight:** how much each kind of token counts (`genre_weight`, `tag_weight`, `decade_weight`).
- Rows are then **L2-normalized** (scaled to length 1).

Example, *Toy Story* (top weights): `tag:pixar` 0.79, `tag:fun` 0.46, `genre:Animation` 0.21, `genre:Children` 0.20, …, `decade:1990s` 0.07.

### Cosine similarity
```
cos(a, b) = a · b / (‖a‖ ‖b‖)  =  a · b      (rows already have length 1)
```
- 1 = same direction (same content), 0 = no tokens in common. TF-IDF weights are never negative, so movie-movie similarity is never below 0.
- `similar(movie)` is one sparse matrix-vector product: `X · x_movie`.
- `explain(a, b)`: the dot product is a sum over tokens of `a_t × b_t`, so each token's share can be read off. *Toy Story* vs *Toy Story 2* = 0.863, of which `tag:pixar` contributes 0.600.

### User profiles
A user's taste is a weighted sum of the vectors of movies they rated:
```
profile_u = Σ_i  w_ui × x_i
```
| `profile` | `w_ui` | Effect |
|---|---|---|
| `centered` (default) | `r_ui − r̄_u` | Above the user's own average pulls toward; below pushes away |
| `liked` | 1 if `r_ui ≥ 4`, else 0 | Only likes count; dislikes are ignored |
| `rating` | `r_ui` | Everything rated pulls toward itself, even a 1-star movie |

`centered` won on val. It's the only option that learns what a user *avoids*.

### Scoring
```
score(u, j) = profile_u · x_j
            = Σ_i  w_ui × (x_i · x_j)
            = Σ_i  w_ui × cos(i, j)
```
The last line is the key insight: **a movie's score is a weighted vote of its similarity to every movie the user rated.** It's close to "item-item neighbours", but with content similarity instead of rating patterns.

- There's no training loop: fitting is counting tokens and adding vectors.
- The profile isn't divided by its length, because scaling a user's whole row by one number doesn't change their ranking.
- The profile is readable (`taste(user)`): user 1 leans toward `genre:Animation` (+4.3) and `genre:Musical` (+3.9), and away from `genre:Fantasy` (−4.3) and `genre:Action` (−4.2).

### Popularity prior
Content alone ranked poorly: val NDCG@10 was 0.013 at best, against 0.061 for popularity. The movies users went on to like had a **median of 29** train ratings. Content's top-10 picks had a **median of 3**. Content finds on-topic movies, and most of the catalog is obscure.

The fix:
```
score = content / max_j |content_j|  +  β × log(1 + n_ratings)
```
- Dividing by the user's largest absolute score puts every user's content scores in [−1, 1], so β means the same thing for every user.
- `log` because the step from 1 to 10 ratings matters more than from 200 to 210.
- β = 0 is pure content. A large β is popularity, with content only breaking ties between equally popular movies.
- At β = 0.5, the most-rated movie gets ≈ 0.5 × log(301) ≈ 2.85, so popularity outweighs content's maximum of 1. That's why the "your own ratings" demo returns the popularity list at β = 0.5.
- A user with no ratings has an all-zero profile, so they get pure popularity, which is a sensible cold-start default.

Rank-percentile blending (convert each model's scores to per-user percentiles, then mix) did worse. Percentiles flatten popularity's long tail: the #1 and #100 most popular movies both land near 1.0.

### Leakage: tags
A user who tagged a movie has seen it. If that movie is in their val or test ratings, its tag carries a trace of the answer. `drop_pairs(tags, val, test)` removes those tags (815 of 3,683). Before trusting a feature, ask: does it know something the model shouldn't?

### Results (test, NDCG@10)

| Model | NDCG@10 | Coverage@10 |
|---|---|---|
| Popularity | 0.0459 | 1.3% |
| Bias (NDCG-tuned) | 0.0502 | 1.0% |
| Content only | 0.0073 | 30% |
| Content + popularity (β = 0.5) | 0.0520 | 2.8% |

The lead over the bias model (~0.002) is within noise.

### Strengths
- **New movies (item cold start):** a movie with no ratings still has genres and tags, so it can be recommended and has neighbours.
- **New users, no retraining:** five ratings give a profile right away. *Matrix* 5, *Inception* 5, *Interstellar* 4.5, *Toy Story* 3, *The Notebook* 1.5 gives *Moon*, *Donnie Darko*, *Arrival* (β = 0).
- **Explainable:** "because you liked movies tagged *sci-fi* and *philosophy*".
- **Similar movies:** serves `GET /movies/{id}/similar` directly.

### Limitations
- **Quality-blind:** tokens say what a movie is about, not whether it's good. A classic and a flop with the same tags score the same.
- **Ties:** movies with identical tokens (same genres and decade, no tags) have identical vectors, so similarity can't separate them. *The Innocents* (1961) has similarity 1.000 to both *Peeping Tom* and *The Collector*.
- **Only as good as the metadata:** 87% of movies have no tags, so they're described by ~2–4 genres and a decade.
- **No surprise:** it recommends more of what you already rate highly and can't learn that sci-fi fans also tend to like a particular crime drama. Collaborative filtering (phase 3) learns those patterns from co-ratings.

### Analogies
- **Librarian:** you say you loved two books about space and philosophy, and the librarian hands you other books shelved under the same subjects. They know the catalogue's labels, not which books readers actually enjoyed.
- **Pandora's Music Genome:** songs are described by hand-labelled attributes (tempo, vocals, instruments) and you get songs with similar attributes. Spotify-style "listeners also played" is collaborative filtering.
- **IDF as "rare words matter":** two articles sharing the word "the" means nothing; sharing "photosynthesis" means a lot. Two movies sharing `genre:Drama` means little; sharing `tag:pixar` means a lot.

## Evaluation
- [Root Mean Square Error (RMSE)](https://c3.ai/resources/glossary/data-science/root-mean-square-error-rmse) 
    - to measure how far the predictions fall from the measured true values using Euclidean distance

- Recall and Precision at K for recommender system: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54

- NCDG metric: https://www.evidentlyai.com/ranking-metrics/ndcg-metric
    - relevance score, k parameter, cumulative gain, discounted gain, normalized DCG
    - intepretation: NDCG vs MAP, NDCG vs MRR


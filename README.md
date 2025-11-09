# Diversity-Aware Sampling

This script picks diverse examples using MMR (Maximum Marginal Relevance). Basically, it finds examples that are both relevant to your query AND different from each other.

## The idea

When picking few-shot examples, you don't want them all to be the same. This fixes that. It:

- First grabs the most relevant example
- Then keeps picking examples that are relevant BUT also different from what's already picked

The `lambda_param` controls the balance:

- 1.0 = just picks most relevant (boring, all similar)
- 0.0 = just picks most diverse (might be irrelevant)
- 0.5 = balanced (usually works well)

## Candidates

Here are some example responses.

```python
candidates = [
    "The movie was fantastic and entertaining",
    "I loved this film, it was amazing",
    "Great cinematography and acting",
    "Terrible movie, waste of time",
    "The plot was confusing and boring",
    "Best film I've seen this year",
    "Not my cup of tea, too slow",
    "Outstanding performances by the cast"
]
```

## What it selected

When I asked "How was the movie?" and set `k` to 3 it picked:

```bash
- The movie was fantastic and entertaining
- Not my cup of tea, too slow
- Best film I've seen this year
```

Notice how it grabbed different perspectives? You get a positive view, a negative one, and an enthusiastic one.

When I asked "How was the movie?" and set `k` to 2 it picked:

```bash
- The movie was fantastic and entertaining
- Not my cup of tea, too slow
```

## Running it

```bash
python mmr-implementation.py
```

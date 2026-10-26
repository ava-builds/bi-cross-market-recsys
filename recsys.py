"""
Cross-market next-destination recommendation on the Booking.com
Multi-Destination Trips dataset (Goldenberg & Levin, SIGIR 2021).

Core idea
---------
Each trip is a sequence of cities. Given the cities booked so far,
recommend the 4 most likely NEXT cities. We compare models that learn
from (a) only the traveller's own booker-country market, (b) all markets
pooled together, and (c) a blend of the two whose weight is chosen on a
validation split (never on the test set).

All models are deliberately simple (counting), so that any difference in
accuracy comes from WHICH DATA a model learns from, not model complexity.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

K = 4  # the Booking.com challenge evaluates the top-4 recommendations


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------
def load(data_dir="data_repo"):
    """Return train bookings, test prefixes (hidden last city removed) and answers."""
    tr = pd.read_csv(f"{data_dir}/train_set.csv", parse_dates=["checkin", "checkout"])
    te = pd.read_csv(f"{data_dir}/test_set.csv", parse_dates=["checkin", "checkout"])
    gt = pd.read_csv(f"{data_dir}/ground_truth.csv")

    tr = tr.sort_values(["utrip_id", "checkin"]).reset_index(drop=True)
    te = te.sort_values(["utrip_id", "checkin"]).reset_index(drop=True)
    te = te[te.city_id != 0]  # city_id 0 = the concealed final booking

    # one row per test trip: country, the visible sequence, and the answer
    trips = (te.groupby("utrip_id")
               .agg(booker_country=("booker_country", "first"),
                    seq=("city_id", list),
                    last_country=("hotel_country", "last"))
               .reset_index()
               .merge(gt.rename(columns={"city_id": "target",
                                         "hotel_country": "target_country"}),
                      on="utrip_id"))
    return tr, trips


def to_trips(bookings):
    """Booking rows -> one row per trip with its city sequence."""
    return (bookings.groupby("utrip_id")
                    .agg(booker_country=("booker_country", "first"),
                         seq=("city_id", list),
                         countries=("hotel_country", list))
                    .reset_index())


def validation_split(train_trips, frac=0.1, seed=0):
    """Hold out the LAST city of a random 10% of training trips (for tuning only)."""
    rng = np.random.default_rng(seed)
    t = train_trips[train_trips.seq.str.len() >= 2]
    mask = rng.random(len(t)) < frac
    val = t[mask].copy()
    fit = pd.concat([train_trips[~train_trips.utrip_id.isin(val.utrip_id)]])
    val["target"] = val.seq.str[-1]
    val["target_country"] = val.countries.str[-1]
    val["last_country"] = val.countries.str[-2]
    val["seq"] = val.seq.str[:-1]
    return fit, val[["utrip_id", "booker_country", "seq", "last_country",
                     "target", "target_country"]]


# ----------------------------------------------------------------------
# 2. Model: first-order transition counts ("people who stayed in A went next to B")
# ----------------------------------------------------------------------
class TransitionModel:
    """
    Learns three count tables from training trips:
      trans[a][b]        how often city b followed city a
      pop_country[c][b]  how often city b was booked in hotel country c
      pop[b]             how often city b was booked overall
    Recommendation for a trip whose last city is a:
      rank cities by trans[a]; if fewer than K, fill with popular cities in the
      same hotel country, then with globally popular cities.
    """

    def fit(self, trips):
        self.trans = defaultdict(Counter)
        self.pop_country = defaultdict(Counter)
        self.pop = Counter()
        self.city_country = {}
        for seq, ctry in zip(trips.seq, trips.countries):
            for a, b in zip(seq[:-1], seq[1:]):
                self.trans[a][b] += 1
            for c, h in zip(seq, ctry):
                self.pop[c] += 1
                self.pop_country[h][c] += 1
                self.city_country[c] = h
        self.global_top = [c for c, _ in self.pop.most_common(50)]
        self.country_top = {h: [c for c, _ in cnt.most_common(50)]
                            for h, cnt in self.pop_country.items()}
        return self

    def trans_dist(self, a):
        """Probability distribution over next cities after city a (empty if unseen)."""
        cnt = self.trans.get(a)
        if not cnt:
            return {}
        n = sum(cnt.values())
        return {b: v / n for b, v in cnt.items()}


def _fill(ranked, seen, *pools):
    out = [c for c in ranked if c not in seen][:K]
    for pool in pools:
        for c in pool:
            if len(out) == K:
                return out
            if c not in out and c not in seen:
                out.append(c)
    return out


def recommend_blend(own, pooled, alpha, seq, last_country):
    """
    Score = alpha * P_own(next | last) + (1 - alpha) * P_pooled(next | last)
      alpha = 1  -> own-market model
      alpha = 0  -> pooled model
    Fallback (when there are < K candidates): popular cities in the same hotel
    country, then globally popular cities. The pure models (alpha = 0 or 1) use
    ONLY their own training data for the fallback too, so the Own model never
    sees other markets' data. The blend uses the higher-weighted model first.
    """
    a = seq[-1]
    seen = set()  # the current city may be recommended again (~9% of next bookings repeat it)
    p_own = own.trans_dist(a) if alpha > 0 else {}
    p_pool = pooled.trans_dist(a) if alpha < 1 else {}
    scores = Counter()
    for b, p in p_own.items():
        scores[b] += alpha * p
    for b, p in p_pool.items():
        scores[b] += (1 - alpha) * p
    ranked = [b for b, _ in scores.most_common(K + 1)]
    if alpha == 1:
        models = [own]
    elif alpha == 0:
        models = [pooled]
    else:
        models = [own, pooled] if alpha >= 0.5 else [pooled, own]
    pools = ([m.country_top.get(last_country, []) for m in models]
             + [m.global_top for m in models])
    return _fill(ranked, seen, *pools)


def recommend_popularity(model, seq):
    return _fill([], set(), model.global_top)


# ----------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------
def score(recs, targets):
    """
    Per-trip Recall@4 and NDCG@4. Each trip has exactly ONE correct next city, so
      Recall@4 = 1 if the answer is in the top 4, else 0
      NDCG@4   = 1 / log2(rank + 1) if the answer is at position `rank` (1..4), else 0
    """
    rec, ndcg = np.zeros(len(targets)), np.zeros(len(targets))
    for i, (r, t) in enumerate(zip(recs, targets)):
        if t in r:
            rank = r.index(t) + 1
            rec[i] = 1.0
            ndcg[i] = 1.0 / np.log2(rank + 1)
    return rec, ndcg


def bootstrap_ci(x, n=1000, seed=0):
    """95% confidence interval of the mean by resampling trips."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_bootstrap_diff(a, b, n=1000, seed=0):
    """Mean of (a - b) and its 95% CI, resampling the SAME trips for both models."""
    d = a - b
    lo, hi = bootstrap_ci(d, n, seed)
    return d.mean(), lo, hi

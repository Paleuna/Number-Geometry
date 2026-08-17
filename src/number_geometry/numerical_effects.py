"""Distance, size, ratio, and 1-D MDS analyses for number representations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.spatial import distance
from sklearn.manifold import MDS


def best_fit_line(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0
    xbar, ybar = x.mean(), y.mean()
    den = np.sum((x - xbar) ** 2)
    b = float(np.sum((x - xbar) * (y - ybar)) / den) if den > 0 else 0.0
    return float(ybar - b * xbar), b


def compute_distance_effect(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    groups, raw = {}, []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = j - i
            sim = 1.0 - distance.cosine(embeddings[i], embeddings[j])
            if np.isfinite(sim):
                groups.setdefault(d, []).append(sim)
                raw.append((i + 1, j + 1, d, float(sim)))
    df = pd.DataFrame([(d, np.mean(groups[d])) for d in sorted(groups)], columns=["distance", "avg_similarity"])
    if len(df) > 1:
        a, b = best_fit_line(df.distance, df.avg_similarity)
        pred = a + b * df.distance.to_numpy()
        ss_res = np.sum((df.avg_similarity.to_numpy() - pred) ** 2)
        ss_tot = np.sum((df.avg_similarity.to_numpy() - df.avg_similarity.mean()) ** 2)
        r2 = 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
    else:
        a = b = r2 = 0.0
    return {"r2": r2, "slope": b, "intercept": a, "data": df, "raw": pd.DataFrame(raw, columns=["i", "j", "distance", "similarity"])}


def compute_size_effect(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = 1.0 - distance.cosine(embeddings[i], embeddings[j])
            if np.isfinite(sim):
                pairs.append((i, j, j - i, float(sim)))
    if not pairs:
        return {"r2": 0.0, "slope": 0.0, "intercept": 0.0, "data": pd.DataFrame(), "raw": pd.DataFrame()}
    sims = np.array([p[3] for p in pairs])
    vmin, vmax = sims.min(), sims.max()
    diff = vmax - vmin if vmax > vmin else 1.0
    raw = pd.DataFrame(
        [(i + 1, i, j, d, sim, (sim - vmin) / diff) for i, j, d, sim in pairs],
        columns=["size_index", "i", "j", "distance", "similarity", "norm_similarity"],
    )
    df = raw.groupby("size_index", as_index=False)["norm_similarity"].mean().rename(columns={"norm_similarity": "avg_normalized_similarity"})
    if len(df) > 1:
        a, b = best_fit_line(df.size_index, df.avg_normalized_similarity)
        pred = a + b * df.size_index.to_numpy()
        y = df.avg_normalized_similarity.to_numpy()
        ss_res, ss_tot = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
        r2 = 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
    else:
        a = b = r2 = 0.0
    return {"r2": r2, "slope": b, "intercept": a, "data": df, "raw": raw}


def compute_ratio_effect(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    rows = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = 1.0 - distance.cosine(embeddings[i], embeddings[j])
            if np.isfinite(sim) and sim != 1.0:
                rows.append(((j + 1) / (i + 1), float(sim)))
    df = pd.DataFrame(rows, columns=["ratio", "similarity"])
    if len(df) < 3:
        return {"r2": 0.0, "params": {"a": 0.0, "b": 0.0, "c": 0.0}, "data": df}
    x, y = df.ratio.to_numpy(), df.similarity.to_numpy()
    try:
        params, _ = scipy.optimize.curve_fit(lambda t, a, b, c: a * np.exp(-b * t) + c, x, y, maxfev=1_000_000)
        a, b, c = map(float, params)
        pred = a * np.exp(-b * x) + c
        ss_res, ss_tot = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
        r2 = 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
    except Exception:
        a = b = c = r2 = 0.0
    return {"r2": r2, "params": {"a": a, "b": b, "c": c}, "data": df}


def compute_mds_analysis(embeddings, numbers):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            d = distance.cosine(embeddings[i], embeddings[j])
            D[i, j] = D[j, i] = d if np.isfinite(d) else 1.0
    try:
        mds = MDS(n_components=1, metric=False, dissimilarity="precomputed", normalized_stress=True, random_state=0)
    except TypeError:  # older scikit-learn
        mds = MDS(n_components=1, metric=False, dissimilarity="precomputed", random_state=0)
    pos = mds.fit_transform(D).ravel()
    if pos[0] > 0:
        pos = -pos
    logs = np.log10(np.asarray(numbers, dtype=float))
    corr = float(np.corrcoef(pos, logs)[0, 1]) if np.std(pos) > 0 else 0.0
    return {"correlation": corr, "stress": float(mds.stress_), "mds_positions": pos, "log_positions": logs}

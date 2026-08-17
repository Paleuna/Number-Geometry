"""Representational alignment analyses used in the paper.

This module contains the reusable numerical part of Procrustes, asymmetric
subspace overlap, SVCCA, PCA dimensionality diagnostics, and permutation
baselines. Plotting intentionally stays in the notebook so each paper figure is
readable in one place.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .preprocessing import available_task_labels, centroid_matrix, task_matrix


def pca_dimension(x: np.ndarray, variance_threshold: float = 0.95, standardize: bool = False) -> int:
    """Smallest PCA dimension that explains at least ``variance_threshold`` variance."""
    x = np.asarray(x, dtype=np.float64)
    if standardize:
        x = StandardScaler().fit_transform(x)
    pca = PCA().fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, variance_threshold) + 1)


def participation_ratio(x: np.ndarray, standardize: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    if standardize:
        x = StandardScaler().fit_transform(x)
    eig = PCA().fit(x).explained_variance_
    denom = np.sum(eig**2)
    return float((np.sum(eig) ** 2) / denom) if denom > 0 else 0.0


def pca_diagnostics(
    raw_df: pd.DataFrame,
    thresholds: Iterable[float] = (0.80, 0.90, 0.95, 0.99),
    fmt: str = "digit",
    standardize: bool = False,
    residualize_number: bool = False,
) -> pd.DataFrame:
    """Inspect task dimensionality before choosing a PCA truncation.

    ``residualize_number=False`` measures the full task representation and is the
    relevant diagnostic for Figure 4. ``True`` reproduces the exploratory
    within-number residual analysis from the original notebook.
    """
    labels = available_task_labels(raw_df, fmt=fmt)
    rows = []
    thresholds = tuple(thresholds)
    for label in labels:
        subset = raw_df[(raw_df["plot_label"] == label) & (raw_df["format"] == fmt)].copy()
        if residualize_number:
            pieces = []
            for _, group in subset.groupby("val", sort=True):
                v = np.stack(group.sort_values("sample_idx")["vector"].to_numpy())
                pieces.append(v - v.mean(axis=0, keepdims=True))
            x = np.vstack(pieces)
        else:
            x = np.stack(subset.sort_values(["val", "sample_idx"])["vector"].to_numpy())
        x_for_pca = StandardScaler().fit_transform(x) if standardize else x
        pca = PCA().fit(x_for_pca)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        eig = pca.explained_variance_
        row = {
            "task": label,
            "format": fmt,
            "n_samples": len(x),
            "participation_ratio": float((eig.sum() ** 2) / np.sum(eig**2)) if np.sum(eig**2) > 0 else 0.0,
            "total_variance": float(eig.sum()),
        }
        for thr in thresholds:
            row[f"k{int(round(thr * 100))}"] = int(np.searchsorted(cumulative, thr) + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(values: Iterable[float]) -> dict:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(np.mean(values))
    if len(values) == 1:
        low = high = mean
    else:
        low, high = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high), "n": int(len(values))}



def procrustes_disparity(x: np.ndarray, y: np.ndarray) -> float:
    """SciPy-equivalent Procrustes disparity, optimized for n_points << hidden_dim.

    ``scipy.spatial.procrustes`` forms a hidden_dim × hidden_dim orthogonal
    problem. Here the same nuclear norm is computed in the much smaller
    point-space, which is important for 9 number centroids in 768+ dimensions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Expected matching 2-D matrices, got {x.shape} and {y.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("Procrustes requires non-degenerate point configurations")
    x, y = x / nx, y / ny
    ux, sx, _ = np.linalg.svd(x, full_matrices=False)
    uy, sy, _ = np.linalg.svd(y, full_matrices=False)
    small = (sx[:, None] * (ux.T @ uy)) * sy[None, :]
    scale = np.linalg.svd(small, compute_uv=False).sum()
    return float(max(0.0, 1.0 - scale**2))

def pairwise_procrustes(centroid_df: pd.DataFrame, fmt: str = "digit", labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or available_task_labels(centroid_df, fmt=fmt)
    manifolds = {label: centroid_matrix(centroid_df, label, fmt) for label in labels}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if manifolds[a].shape != manifolds[b].shape:
                mat[i, j] = np.nan
            else:
                mat[i, j] = procrustes_disparity(manifolds[a], manifolds[b])
    return pd.DataFrame(mat, index=labels, columns=labels)


def procrustes_permutation_baseline(
    centroid_df: pd.DataFrame,
    fmt: str = "digit",
    labels: list[str] | None = None,
    n_permutations: int = 1000,
    random_state: int = 0,
) -> dict:
    """Observed pairwise disparity vs a number-identity shuffled null."""
    labels = labels or available_task_labels(centroid_df, fmt=fmt)
    manifolds = {label: centroid_matrix(centroid_df, label, fmt) for label in labels}
    observed = [procrustes_disparity(manifolds[a], manifolds[b]) for a, b in combinations(labels, 2)]
    rng = np.random.default_rng(random_state)
    null = []
    for _ in range(n_permutations):
        a, b = rng.choice(labels, size=2, replace=True)
        x, y = manifolds[a], manifolds[b]
        y_perm = y[rng.permutation(len(y))]
        null.append(procrustes_disparity(x, y_perm))
    return {"observed": _summary(observed), "null": _summary(null), "observed_values": np.asarray(observed), "null_values": np.asarray(null)}



def _fit_pca_subspace(source: np.ndarray, variance_threshold: float, n_components: int | None):
    source = np.asarray(source, dtype=np.float64)
    if n_components is None:
        return PCA(n_components=variance_threshold, svd_solver="full").fit(source)
    max_k = min(source.shape[0] - 1, source.shape[1])
    return PCA(n_components=min(int(n_components), max_k)).fit(source)


def subspace_overlap(
    source: np.ndarray,
    target: np.ndarray,
    variance_threshold: float = 0.95,
    n_components: int | None = None,
) -> float:
    """Fraction of target variance explained by the source PCA subspace.

    This is asymmetric: ``source <- target`` need not equal ``target <- source``.
    When ``n_components`` is omitted, the source task keeps the smallest number
    of PCs explaining ``variance_threshold`` of its own variance.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pca = _fit_pca_subspace(source, variance_threshold, n_components)
    centered_target = target - target.mean(axis=0, keepdims=True)
    projected = centered_target @ pca.components_.T
    explained = np.var(projected, axis=0, ddof=0).sum()
    total = np.var(target, axis=0, ddof=0).sum()
    return float(explained / total) if total > 0 else np.nan


def pairwise_subspace_overlap(
    raw_df: pd.DataFrame,
    fmt: str = "digit",
    labels: list[str] | None = None,
    variance_threshold: float = 0.95,
    n_components: int | None = None,
) -> pd.DataFrame:
    """Efficient pairwise overlap: each source PCA is fit once."""
    labels = labels or available_task_labels(raw_df, fmt=fmt)
    mats = {label: task_matrix(raw_df, label, fmt).astype(np.float64) for label in labels}
    pcas = {label: _fit_pca_subspace(mats[label], variance_threshold, n_components) for label in labels}
    centered = {label: mats[label] - mats[label].mean(axis=0, keepdims=True) for label in labels}
    total_var = {label: np.var(mats[label], axis=0, ddof=0).sum() for label in labels}
    out = np.zeros((len(labels), len(labels)), dtype=float)
    for i, source in enumerate(labels):
        components = pcas[source].components_
        for j, target in enumerate(labels):
            projected = centered[target] @ components.T
            explained = np.var(projected, axis=0, ddof=0).sum()
            out[i, j] = explained / total_var[target] if total_var[target] > 0 else np.nan
    return pd.DataFrame(out, index=labels, columns=labels)


def _pca_scores(
    x: np.ndarray,
    variance_threshold: float,
    n_components: int | None,
    standardize: bool,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if standardize:
        x = StandardScaler().fit_transform(x)
    if n_components is None:
        pca = PCA(n_components=variance_threshold, svd_solver="full")
    else:
        max_k = min(len(x) - 1, x.shape[1])
        pca = PCA(n_components=min(int(n_components), max_k))
    return pca.fit_transform(x)


def _cca_mean(z_x: np.ndarray, z_y: np.ndarray, max_iter: int = 20000) -> tuple[float, np.ndarray]:
    if len(z_x) != len(z_y):
        raise ValueError(f"SVCCA requires paired observations; got {len(z_x)} and {len(z_y)}")
    k = min(z_x.shape[1], z_y.shape[1], len(z_x) - 1)
    x, y = z_x[:, :k], z_y[:, :k]
    cca = CCA(n_components=k, max_iter=max_iter)
    c_x, c_y = cca.fit_transform(x, y)
    correlations = np.array([np.corrcoef(c_x[:, i], c_y[:, i])[0, 1] for i in range(k)], dtype=float)
    correlations = np.abs(correlations[np.isfinite(correlations)])
    return float(np.mean(correlations)), correlations


def svcca_similarity(
    x: np.ndarray,
    y: np.ndarray,
    variance_threshold: float = 0.95,
    n_components: int | None = None,
    standardize: bool = True,
    max_iter: int = 20000,
) -> float:
    """Mean canonical correlation after PCA truncation.

    The default keeps 95% variance separately for each task. ``standardize=True``
    matches the feature-standardization step used in the submitted analysis.
    """
    z_x = _pca_scores(x, variance_threshold, n_components, standardize)
    z_y = _pca_scores(y, variance_threshold, n_components, standardize)
    return _cca_mean(z_x, z_y, max_iter=max_iter)[0]


def pairwise_svcca(
    raw_df: pd.DataFrame,
    fmt: str = "digit",
    labels: list[str] | None = None,
    variance_threshold: float = 0.95,
    n_components: int | None = None,
    standardize: bool = True,
    max_iter: int = 20000,
) -> pd.DataFrame:
    """Pairwise SVCCA with PCA scores cached once per task."""
    labels = labels or available_task_labels(raw_df, fmt=fmt)
    mats = {label: task_matrix(raw_df, label, fmt) for label in labels}
    scores = {
        label: _pca_scores(mats[label], variance_threshold, n_components, standardize)
        for label in labels
    }
    out = np.zeros((len(labels), len(labels)), dtype=float)
    for i, a in enumerate(labels):
        for j in range(i, len(labels)):
            b = labels[j]
            score = _cca_mean(scores[a], scores[b], max_iter=max_iter)[0]
            out[i, j] = out[j, i] = score
    return pd.DataFrame(out, index=labels, columns=labels)


def compare_pc_strategies(
    raw_df: pd.DataFrame,
    fmt: str = "digit",
    fixed_k: int = 17,
    variance_threshold: float = 0.95,
    svcca_standardize: bool = True,
) -> pd.DataFrame:
    """Small robustness diagnostic: fixed-k versus variance-based Figure 4 metrics."""
    overlap_fixed = pairwise_subspace_overlap(raw_df, fmt=fmt, n_components=fixed_k)
    overlap_adapt = pairwise_subspace_overlap(raw_df, fmt=fmt, variance_threshold=variance_threshold)
    svcca_fixed = pairwise_svcca(raw_df, fmt=fmt, n_components=fixed_k, standardize=svcca_standardize)
    svcca_adapt = pairwise_svcca(raw_df, fmt=fmt, variance_threshold=variance_threshold, standardize=svcca_standardize)

    def matrix_summary(name: str, a: pd.DataFrame, b: pd.DataFrame, symmetric: bool) -> dict:
        n = len(a)
        mask = np.triu(np.ones((n, n), dtype=bool), 1) if symmetric else ~np.eye(n, dtype=bool)
        av, bv = a.to_numpy()[mask], b.to_numpy()[mask]
        return {
            "metric": name,
            "fixed_mean": float(np.mean(av)),
            "adaptive_mean": float(np.mean(bv)),
            "mean_abs_difference": float(np.mean(np.abs(av - bv))),
            "matrix_correlation": float(np.corrcoef(av, bv)[0, 1]),
        }

    return pd.DataFrame([
        matrix_summary("Subspace overlap", overlap_fixed, overlap_adapt, symmetric=False),
        matrix_summary("SVCCA", svcca_fixed, svcca_adapt, symmetric=True),
    ])

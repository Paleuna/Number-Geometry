"""Loading and preprocessing activation pickles used in the paper.

The activation pickle format is intentionally kept compatible with the original
experiments: keys are ``(task, subset, number, format)`` and values are arrays
with shape ``[samples, layers, hidden_dim]``.
"""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_ORDER = [
    "Natural Sentence",
    "Pseudo Sentence",
    "Quantity",
    "Addition (pre)",
    "Addition (post)",
    "Multiplication (pre)",
    "Multiplication (post)",
    "Parity (odd)",
    "Parity (even)",
    "Prime (prime)",
    "Prime (composite)",
    "Comparison (greater)",
    "Comparison (smaller)",
    "Successor",
    "Predecessor",
]


PAPER_TASK_ORDER = [
    "Quantity",
    "Addition (pre)",
    "Addition (post)",
    "Multiplication (pre)",
    "Multiplication (post)",
    "Parity",
    "Prime",
    "Comparison (greater)",
    "Comparison (smaller)",
    "Successor",
    "Predecessor",
]


def get_plot_label(task: str, subset: str, split_subtasks: bool = True) -> str:
    """Return the human-readable label used throughout the analysis notebook."""
    mapping = {
        "real_sample": "Natural Sentence",
        "real_insert": "Pseudo Sentence",
        "quantity": "Quantity",
        "successor": "Successor",
        "predecessor": "Predecessor",
    }
    if task in mapping:
        return mapping[task]
    if task == "addition":
        return "Addition (pre)" if "pre" in subset else "Addition (post)"
    if task == "multiplication":
        return "Multiplication (pre)" if "pre" in subset else "Multiplication (post)"
    if task == "comparison":
        return "Comparison (greater)" if subset == "greater" else "Comparison (smaller)"
    if task in {"parity", "prime"}:
        if split_subtasks:
            return f"{task.capitalize()} ({subset})"
        return task.capitalize()
    return f"{task.capitalize()} ({subset})"


def layer_index_from_fraction(n_layers: int, layer_fraction: float = 0.75) -> int:
    """Map a depth fraction to the same layer index used in the submitted code.

    The activation files contain hidden layers only (the embedding layer was
    skipped during extraction). For 12 BERT hidden layers and ``0.75`` this
    returns index 9, matching the original notebook.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    if not 0 <= layer_fraction < 1:
        raise ValueError("layer_fraction must satisfy 0 <= fraction < 1")
    return min(int(n_layers * layer_fraction), n_layers - 1)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def load_activation_pickle(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Expected a non-empty dict activation pickle")
    return data


def build_representation_tables(
    path: str | Path,
    layer_fraction: float = 0.75,
    split_subtasks: bool = True,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one model and return raw-sample and centroid tables.

    Returns
    -------
    raw_df
        One row per sentence/sample. Important columns are ``task``, ``subset``,
        ``val``, ``format``, ``sample_idx``, ``plot_label`` and ``vector``.
    centroid_df
        One row per task/subset/number/format centroid. Centroids are re-normalized
        to unit length after averaging, matching the Figure 1 preprocessing.
    """
    raw = load_activation_pickle(path)
    first = next(iter(raw.values()))
    if np.asarray(first).ndim != 3:
        raise ValueError("Each pickle value must have shape [samples, layers, hidden_dim]")
    layer_idx = layer_index_from_fraction(first.shape[1], layer_fraction)

    rows: list[dict] = []
    for key, arr in raw.items():
        if len(key) != 4:
            raise ValueError(f"Unexpected activation key: {key!r}")
        task, subset, val, fmt = key
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected activation shape for {key}: {arr.shape}")
        if layer_idx >= arr.shape[1]:
            raise ValueError(f"Layer {layer_idx} unavailable for {key}: {arr.shape}")
        vectors = np.asarray(arr[:, layer_idx, :], dtype=np.float32)
        if normalize:
            vectors = l2_normalize_rows(vectors)
        label = get_plot_label(task, subset, split_subtasks=split_subtasks)
        for sample_idx, vec in enumerate(vectors):
            rows.append(
                {
                    "task": task,
                    "subset": subset,
                    "plot_label": label,
                    "val": int(val),
                    "format": fmt,
                    "sample_idx": int(sample_idx),
                    "vector": vec,
                }
            )

    raw_df = pd.DataFrame(rows)
    centroid_rows: list[dict] = []
    group_cols = ["task", "subset", "plot_label", "val", "format"]
    for keys, group in raw_df.groupby(group_cols, sort=False):
        avg = np.mean(np.stack(group["vector"].to_numpy()), axis=0)
        if normalize:
            avg = avg / (np.linalg.norm(avg) + 1e-10)
        record = dict(zip(group_cols, keys))
        record["vector"] = avg.astype(np.float32)
        centroid_rows.append(record)

    centroid_df = pd.DataFrame(centroid_rows)
    raw_df.attrs.update({"layer_idx": layer_idx, "layer_fraction": layer_fraction, "source": str(path)})
    centroid_df.attrs.update(raw_df.attrs)
    return raw_df, centroid_df


def available_task_labels(
    df: pd.DataFrame,
    fmt: str = "digit",
    exclude_sentences: bool = True,
    preferred_order: Iterable[str] | None = None,
) -> list[str]:
    labels = list(df.loc[df["format"] == fmt, "plot_label"].unique())
    if exclude_sentences:
        labels = [x for x in labels if "Sentence" not in x]
    order = list(preferred_order) if preferred_order is not None else PAPER_TASK_ORDER
    ordered = [x for x in order if x in labels]
    ordered += [x for x in labels if x not in ordered]
    return ordered


def task_matrix(df: pd.DataFrame, label: str, fmt: str = "digit") -> np.ndarray:
    """Return raw samples ordered explicitly by number and sentence index."""
    subset = df[(df["plot_label"] == label) & (df["format"] == fmt)].sort_values(
        ["val", "sample_idx"], kind="stable"
    )
    if subset.empty:
        raise KeyError(f"No data for task={label!r}, format={fmt!r}")
    return np.stack(subset["vector"].to_numpy())


def centroid_matrix(df: pd.DataFrame, label: str, fmt: str = "digit") -> np.ndarray:
    subset = df[(df["plot_label"] == label) & (df["format"] == fmt)].sort_values("val")
    if subset.empty:
        raise KeyError(f"No centroid data for task={label!r}, format={fmt!r}")
    return np.stack(subset["vector"].to_numpy())


def relabel_subtasks(df: pd.DataFrame, split_subtasks: bool) -> pd.DataFrame:
    """Return a copy with ``plot_label`` recomputed from task/subset metadata."""
    out = df.copy()
    out["plot_label"] = [
        get_plot_label(task, subset, split_subtasks=split_subtasks)
        for task, subset in zip(out["task"], out["subset"])
    ]
    out.attrs.update(df.attrs)
    return out

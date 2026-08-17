import pickle
import numpy as np

from number_geometry.preprocessing import build_representation_tables, layer_index_from_fraction, relabel_subtasks


def test_layer_index_matches_submitted_rule():
    assert layer_index_from_fraction(12, 0.75) == 9


def test_build_tables_and_sample_index(tmp_path):
    rng = np.random.default_rng(0)
    data = {
        ("quantity", "control", val, "digit"): rng.normal(size=(5, 12, 8)).astype(np.float32)
        for val in range(1, 10)
    }
    p = tmp_path / "toy.pkl"
    with p.open("wb") as f:
        pickle.dump(data, f)
    raw, cent = build_representation_tables(p, layer_fraction=0.75)
    assert len(raw) == 45
    assert len(cent) == 9
    assert raw.sample_idx.max() == 4
    norms = np.linalg.norm(np.stack(raw.vector), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_relabel_merges_property_subtasks():
    import pandas as pd
    df = pd.DataFrame({"task": ["parity", "parity"], "subset": ["odd", "even"], "plot_label": ["Parity (odd)", "Parity (even)"]})
    merged = relabel_subtasks(df, split_subtasks=False)
    assert merged.plot_label.tolist() == ["Parity", "Parity"]

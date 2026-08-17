# The Representational Geometry of Number
CogSci 2026

[![Paper](https://img.shields.io/badge/Paper-PDF-brightred)](https://arxiv.org/abs/2602.06843)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-blue)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**TL;DR.** We find that number representations in language models preserve stable **between-concept geometric relations** across tasks while being embedded in distinct task-specific subspaces. Despite this separation, these task-specific representations remain largely related through linear transformations. Together, these results suggest a representational scheme in which shared conceptual structure is preserved across task-dependent coordinate systems, allowing models to combine relational stability with functional flexibility.

This repository is intentionally notebook-first. Use [`data_processing.ipynb`](notebooks/data_processing.ipynb) to extract model activations and [`representation_analysis.ipynb`](notebooks/representation_analysis.ipynb) to reproduce the analyses and figures. Reusable helper functions are provided as plain Python files in [`src/`](src/).

Because the code has been refactored for clarity and reproducibility, some numerical values may differ slightly from those reported in the paper. These differences do not affect the qualitative patterns or main conclusions.

## File layout

```text
.
├── README.md
├── requirements.txt
├── activations/                  # spatial_analysis_*.pkl (not tracked by git)
├── corpus_data/                  # natural / pseudo-sentence data
├── src/number_geometry/
│   ├── stimuli.py                # task and corpus stimuli
│   ├── models.py                 # Hugging Face model loading
│   ├── extraction.py             # target-token activation extraction
│   ├── preprocessing.py          # activation -> analysis tables
│   ├── numerical_effects.py      # distance / size / ratio effects
│   └── representation.py         # Procrustes / subspace overlap / SVCCA
├── notebooks/
│   ├── data_processing.ipynb
│   └── representation_analysis.ipynb
├── outputs/                      # generated figures
└── tests/
```

## How to run

### Colab (recommended) or local environment

Clone the repository and move into it:

```python
!git clone https://github.com/Paleuna/Number-Geometry.git
%cd Number-Geometry
```

For a local environment, install the dependencies and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter lab
```

If you **already have activation files**, place them in:

```text
activations/spatial_analysis_bert.pkl
activations/spatial_analysis_qwen2.5-math.pkl
```

Then run:

```text
notebooks/representation_analysis.ipynb
```

If the activation files are stored in Google Drive, edit `ACTIVATION_DIR` in the notebook accordingly.

If you want to **regenerate activations**, run:

```text
notebooks/data_processing.ipynb
```

---

## Mathematical guide to the analyses

The paper does not use any single metric to determine which task is “most similar” to another. Instead, the analyses are complementary. Together, they ask:

- whether numerical **relations** are preserved,
- whether task representations occupy the same **subspace**, and
- whether representations in distinct subspaces remain **linearly alignable**.

### 1. Mental Number Line (MNL) effects

The Mental Number Line hypothesis predicts a compressed representational geometry of numerical magnitude. We test three classic signatures using pairwise similarity between number representations.

- **Distance effect.** Nearby numbers (for example, 4 and 5) are represented more similarly than distant numbers (4 and 9). Similarity therefore decreases as numerical distance increases.
- **Size effect.** For the same absolute distance, larger numbers are less separated than smaller numbers: the representational gap between 8 and 9 is smaller than the gap between 1 and 2. This is consistent with compression at larger magnitudes.
- **Ratio effect.** Discriminability depends on relative rather than only absolute distance. Pairs with ratios closer to 1 are represented more similarly, another signature of approximately logarithmic compression.

Taken together, these effects test whether the **within-task relational organization** of the numbers 1--9 resembles the geometry expected from human numerical cognition.

### 2. Procrustes analysis: is the relational shape preserved across tasks?

For two configurations, \(X\) and \(Y\), Procrustes analysis finds the translation, rotation or reflection, and uniform scaling that best align them:

```math
M^2
=
\min_{s,R,t}
\left\|
sXR + \mathbf{1}t^\top - Y
\right\|_F^2,
\qquad
R^\top R = I
```

The resulting \(M^2\) is the **Procrustes disparity**. A small disparity means that two tasks may place number representations in different locations and orientations while preserving nearly the same relational configuration.

The key intuition is that Procrustes deliberately factors out coordinate choices. It does not ask whether two tasks use the same axes or occupy the same region of representation space. Instead, it asks whether the **shape formed by the numerical concepts relative to one another** is preserved.

The number-identity permutation baseline provides a null comparison by destroying the correct correspondence between 1--9 while leaving the representation distributions otherwise intact.

### 3. Subspace overlap: do tasks use the same representational directions?

Let

```math
V_A = [v_{A,1}, \ldots, v_{A,k}]
```

contain the top-\(k\) principal directions of task \(A\), and let \(X_B\) denote the centered representations of task \(B\). We measure how much of the variance in \(B\) can be captured using the principal subspace learned from \(A\):

```math
\mathrm{Overlap}(A \leftarrow B)
=
\frac{
\sum_{i=1}^{k} \mathrm{Var}(X_B v_{A,i})
}{
\mathrm{Var}(X_B)
}
=
\frac{
\lVert X_B V_A \rVert_F^2
}{
\lVert X_B \rVert_F^2
}
```

This quantity is **asymmetric**:

```math
\mathrm{Overlap}(A \leftarrow B)
\neq
\mathrm{Overlap}(B \leftarrow A)
```

in general.

A low overlap means that the dominant directions used by one task explain relatively little of the variance used by another. Thus, two tasks can preserve similar numerical relationships while embedding them in different representational directions.

### 4. SVCCA: can distinct task representations be linearly aligned?

Singular Vector Canonical Correlation Analysis (SVCCA) first reduces each representation with PCA and then performs Canonical Correlation Analysis (CCA). For PCA-reduced representations \(Z_A\) and \(Z_B\), CCA finds pairs of linear projections that maximize their correlation:

```math
\rho_i
=
\max_{w_{A,i}, w_{B,i}}
\mathrm{Corr}
\left(
Z_A w_{A,i},
Z_B w_{B,i}
\right)
```

Successive canonical directions are constrained to capture new, non-redundant correlations. We summarize the overall alignment using the mean canonical correlation:

```math
\bar{\rho}
=
\frac{1}{k}
\sum_{i=1}^{k}
\rho_i
```

Subspace overlap asks whether two tasks use the **same directions in the original representation space**.

SVCCA asks a different question: after allowing a separate linear change of coordinates for each task, can their sample-wise representational structure still be brought into correspondence?

High SVCCA together with low subspace overlap therefore supports **linear transformability without shared coordinates**.

---

## How to read the paper's main claim

The intended argument is a sequence of complementary results, not a ranking of task pairs.

1. **MNL effects establish within-task numerical structure.**  
   Within individual tasks, the relations among numbers follow a magnitude geometry consistent with the mental number line.

2. **Procrustes establishes cross-task relational stability.**  
   Across tasks, this relational shape remains similar even after allowing changes in orientation, translation, and scale. Together with the MNL results, this provides the primary evidence for a **stable relational structure**.

3. **Subspace overlap establishes task-specific separation.**  
   Shared relational structure does **not** imply that tasks reuse the same representational directions. Many task pairs occupy substantially distinct subspaces.

4. **SVCCA establishes linear transformability.**  
   Despite this directional separation, much of the cross-task correspondence can still be recovered through linear transformations.

The conclusion is therefore not that one particular task is “closest” to another. The claim is that numerical concepts exhibit **stable relations in different coordinates**: number geometry is preserved across task-dependent representations, while task-specific subspaces provide flexibility and separation.

In this interpretation, **MNL effects and Procrustes** provide the primary evidence for a shared numerical geometry, **Subspace Overlap** establishes task-specific separation, and **SVCCA** shows that the resulting representations remain largely linearly transformable.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{hu2026representational,
  title={The Representational Geometry of Number},
  author={Hu, Zhimin and Niu, Lanhao and Varma, Sashank},
  journal={arXiv preprint arXiv:2602.06843},
  year={2026}
}
```

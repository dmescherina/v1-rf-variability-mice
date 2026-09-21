# Variabilities in the Shapes of the Receptive Fields of Simple Cells in Mice

Code and derived data for the paper:

> Meshcherina, D., Auffarth, B., & Lindeberg, T. (2026). *Variabilities in the shapes of
> the receptive fields of simple cells in mice*. PLOS Computational Biology.
> https://doi.org/XXXXXXXX

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## What this repository contains

This study characterises the spatial structure of receptive fields in mouse primary visual cortex (V1) using the Allen Brain Observatory Visual Coding 2-photon dataset. Receptive fields are reconstructed from locally sparse noise stimuli via ridge regression, then fitted with Gaussian-derivative models of orders m = 0, 1, and 2, following Lindeberg's (2021, 2024) scale-space framework. We identify 31 first-order (m = 1) simple cells from a population of 1,017 well-fitted neurons across 17 experiment containers, characterise their receptive field shapes, and compare the Gaussian-derivative description against Gabor fits. The analysis covers scale (σ), elongation (κ), and differentiation direction (φ) and tests whether these parameters show spatial organisation within the imaging plane.

This repository contains:

- **Core library** (`src/rf_analysis/`) — RF reconstruction and model fitting
- **Analysis notebooks** (`notebooks/`) — the full pipeline from raw data to figures
- **Derived data** (`derived_data/`) — fitted parameters and the 31-cell dataset; sufficient to reproduce all figures without downloading the raw imaging data
- **Manuscript figures** (`outputs/figures/`) — publication-ready PDF, SVG, and PNG

---

## Requirements

Python 3.11. Install dependencies with conda (recommended) or pip:

```bash
# conda
conda env create -f environment.yml
conda activate v1-rf-shapes

# pip
pip install -r requirements.txt
```

The Allen SDK is installed via pip inside the conda environment. The raw NWB files (~22 GB) are fetched on demand by notebook 01; see [Getting the Allen data](#getting-the-allen-data) below.

---

## Reproducing the figures

### From derived data only (no download required)

All six manuscript figures and the supplementary PDF can be reproduced directly from the data in `derived_data/`:

```bash
cd notebooks/
jupyter lab
```

Run in order:

| Notebook | Produces |
|----------|----------|
| `05_population_stats.ipynb` | Key statistics reported in the paper |
| `06_figures.ipynb` | `outputs/figures/fig02_*` … `fig07_*` |
| `07_supplementary_pdf.ipynb` | `derived_data/m1_cells/supplementary_rf_gallery_m1_31.pdf` |

### Full pipeline (from raw Allen data)

To reproduce everything from scratch, including RF reconstruction and model fitting:

| Notebook | Input | Output |
|----------|-------|--------|
| `01_download_allen_data.ipynb` | Allen API | `data/cache/` NWB files |
| `02_rf_population.ipynb` | NWB files | `derived_data/population/` CSVs |
| `03_m1_gallery.ipynb` | NWB files + population CSVs | `derived_data/m1_cells/m1_neuron_dataset.pkl` |
| `04_gabor_comparison.ipynb` | NWB files + m1 dataset | `derived_data/m1_cells/gabor_vs_gd_m1_31_v3.csv` |
| `05_population_stats.ipynb` | population CSVs + m1 dataset | statistics |
| `06_figures.ipynb` | all derived data | `outputs/figures/` |
| `07_supplementary_pdf.ipynb` | population CSVs + m1 dataset | supplementary PDF |

All paths are resolved relative to the repository root; notebooks can be run from either the `notebooks/` directory or the repo root.

---

## Getting the Allen data

The raw imaging data are publicly available through the Allen Brain Observatory and are not redistributed here. Notebook `01_download_allen_data.ipynb` downloads the 51 NWB files needed for this analysis (~22 GB) using the AllenSDK:

```python
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file='data/cache/manifest.json')
```

The 17 experiment containers used are listed in `data/experiment_lists/target_containers_metadata.csv`. AllenSDK documentation: https://allensdk.readthedocs.io/

---

## Derived data

All derived data are in `derived_data/` and are sufficient to reproduce figures and statistics without the raw NWB files.

```
derived_data/
├── population/
│   ├── rf_params_order_v2_all_containers.csv   # 1,017 neurons; fitted parameters + m_final
│   └── rf_params_order_v2_container_*.csv      # per-container files (17 containers)
├── m1_cells/
│   ├── m1_neuron_dataset.pkl                   # 31 verified m=1 simple cells (RF maps + params)
│   ├── m1_neuron_dataset.csv                   # same, tabular
│   └── gabor_vs_gd_m1_31_v3.csv               # Gabor vs Gaussian-derivative comparison
└── review_judgements/                          # manual verification records (input to notebook 02)
```

**Column glossary for `rf_params_order_v2_all_containers.csv`:**

| Column | Description |
|--------|-------------|
| `cell_id` | Allen cell specimen ID |
| `container_id` | Allen experiment container ID |
| `derivative_order` | Best-fitting model order (0, 1, or 2) |
| `m_final` | Final order after manual review (0, 1, or 2) |
| `manually_verified` | Whether the cell underwent manual inspection |
| `sigma` | Geometric-mean receptive field scale (degrees) |
| `kappa` | Elongation, σ_major / σ_minor ≥ 1 |
| `theta` | Envelope orientation, major axis angle (degrees, [0°, 180°)) |
| `phi` | Differentiation direction from lobe geometry (degrees, [0°, 360°) for m=1) |
| `theta_hybrid` | φ for m=1, θ for m=0 (recommended orientation column) |
| `r_squared` | Goodness of fit of the selected model |
| `cortex_x_um`, `cortex_y_um` | ROI centroid position within the imaging plane (µm) |

---

## Figure 1

Figure 1 is a schematic overview produced by `fig01_overview.py` (not yet committed; see
`outputs/figures/` for the current draft). All data figures (2–7) are produced by
`notebooks/06_figures.ipynb`.

---

## Citation

```bibtex
@article{meshcherina2026rf,
  author  = {Meshcherina, Daria and Auffarth, Ben and Lindeberg, Tony},
  title   = {Variabilities in the shapes of the receptive fields of simple cells in mice},
  journal = {PLOS Computational Biology},
  year    = {2026},
  doi     = {XXXXXXXX}
}
```

**Zenodo archive** (code + derived data): https://doi.org/10.5281/zenodo.XXXXXXX

---

## Licence

MIT — see [LICENSE](LICENSE).

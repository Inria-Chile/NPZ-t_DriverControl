# Beyond the Critical Depth: NPZ-T Global Persistence Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Powered by Pangeo](https://img.shields.io/badge/Powered%20by-Pangeo-orange.svg)](https://pangeo.io/)

This repository contains the official implementation of the **NPZ-T (Nutrient-Phytoplankton-Zooplankton-Temperature)** diagnostic framework. The model applies **non-autonomous dynamical systems theory** to CMIP6 climate projections to identify global phytoplankton persistence thresholds and habitat regime shifts.

## 🔬 Scientific Context

Traditional "Critical Depth" theories often neglect the non-linear coupling between seasonal mixing and metabolic temperature dependence. This project implements a diagnostic tool to calculate the **Critical Nutrient Threshold** ($\gamma_{crit}$), identifying the transition from mixing-driven to metabolic-driven ecological regimes.

The core stability condition is defined by the annual integral of the net growth rate:
$$\frac{1}{T} \int_{0}^{T} \left[ \mu(t, T, I) - m(t, T) - g(t, Z) - \sigma(t) \right] dt = 0$$
where $\sigma(t)$ represents the dilution rate driven by Mixed Layer Depth (MLD) dynamics.

## 🛠 Repository Structure

* **`npz_t_engine.py`**: The mathematical core. Implements Thermodynamic Performance Curves (TPC), linearized Floquet stability analysis, and the calculation of critical bifurcation points.
* **`data_processing.py`**: High-performance data pipeline. Connects to the **Pangeo CMIP6 Cloud** via `intake-esm`, manages **Dask** distributed clusters, and performs `xMIP` preprocessing for models (e.g., GFDL-ESM4, ACCESS-CM2).
* **`run_plots.py`**: Visualization suite and LaTeX exporter. Generates global maps using `cartopy` and automates the generation of results for the manuscript.
* **`data/`**: Local storage for processed `.pkl` files and exported LaTeX tables (ignored by git).

## 🚀 Getting Started

### 1. Environment & Dependencies
This project requires `ESMF` for regridding and specific climate science libraries. 

**System requirements (Ubuntu/Debian):**
```bash
sudo apt install gfortran gcc make perl libblas-dev liblapack-dev libyaml-cpp-dev openmpi-bin libopenmpi-dev

**Python dependencies:**

Ensure you have intake-esm, xarray, xesmf, dask, swifter, and datatree installed.

**2. Configuration**

The data_processing.py script automatically locates your ESMFMKFILE if it follows the standard installation path: ~/git/esmf/. If you are using a custom path, please set the environment variable manually.

**3. Execution Pipeline**

    Process Data: Download CMIP6 vectors and run the NPZ-T engine globally:
    Bash

    python data_processing.py

    Generate Figures & Stats: Execute the plotting script to generate maps and LaTeX outputs:
    Bash

    python run_plots.py

📊 Automated LaTeX Integration

To ensure 100% consistency between the code and the manuscript, this framework exports two types of LaTeX files to data/latex/:

    habitat_vars.tex: Automatically generates \newcommand macros (e.g., \GFDLViableNicheArea). Use these in your .tex document to cite exact values that update automatically if the model parameters change.

    habitat_table.tex: A production-ready table with professional formatting containing global area statistics and coverage percentages.

## 📜 Citation

If you use this framework or the NPZ-T model in your research, please cite the following manuscript:

```bibtex
@article{Neto2025NPZT,
  title={Beyond the Critical Depth: The Metabolic and Physical Drivers of Phytoplankton Persistence in a Changing Ocean},
  author={Neto, Matías and Marquet, Pablo A. and Freilich, Mara and Martí, Luis and Sanchez-Pi, Nayat},
  journal={ArXiv (Preprint)},
  year={2026},
  note={Under review}
}


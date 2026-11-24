# Emerging Thermal-Metabolic Control of Phytoplankton Annual Cycles (NPZ-t)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Powered by Pangeo](https://img.shields.io/badge/Powered%20by-Pangeo-orange.svg)](https://pangeo.io/)

Official implementation of the **NPZ-t (Nutrient-Phytoplankton-Zooplankton-Temperature)** diagnostic framework. The model applies non-autonomous periodic dynamical systems theory to CMIP6 climate projections (GFDL-ESM4, SSP5-8.5) to quantify the physical and metabolic drivers governing phytoplankton annual cycles.

## Scientific Context

This repository implements a diagnostic framework to evaluate the **Critical Nutrient Saturation Ratio** ($\gamma_{\text{crit}}$) and diagnose global regime shifts between physical-mixing, thermal-metabolic, and light-amplitude controls. By linearizing the non-autonomous system around the extinction equilibrium ($E_0$), the stability boundary is derived algebraically via the phytoplankton Floquet exponent ($\lambda_P$), followed by a first-order Taylor sensitivity decomposition across environmental anomalies ($\Delta T$, $\Delta H$, $\Delta I$).

## Repository Structure

- **`npz_t_engine.py`**: Mathematical core. Implements Thermodynamic Performance Curves (TPC), Floquet stability thresholds ($\lambda_P$, $\gamma_{\text{crit}}$), and trivariate sensitivity gradients.
- **`data_processing.py`**: Data pipeline. Connects to the **Pangeo CMIP6 Cloud** via `intake-esm`, manages **Dask** distributed clusters, and performs `xMIP` preprocessing.
- **`run_plots.py`**: Visualization suite and LaTeX exporter. Generates global driver maps and 1D Kernel Density Estimates (KDE) using `cartopy`, automating numerical outputs for the manuscript.
- **`data/`**: Local storage for processed `.pkl` files and exported LaTeX tables (ignored by git).

## 🚀 Getting Started

### 1. Environment & Dependencies
This project requires `ESMF` for regridding and specific climate science libraries. 

**System requirements**:
Install required build tools according to your operating system (see [9.3 System Specific Information](https://earthsystemmodeling.org/docs/nightly/develop/ESMF_usrdoc/node10.html#SECTION000103000000000000000)).


**Ubuntu Linux:**
```bash
sudo apt install gfortran gcc make perl libblas-dev liblapack-dev libyaml-cpp-dev openmpi-bin libopenmpi-dev openmpi-common
```
**macOS**
```bash
brew install gfortran gcc gcc@9 openmpi perl lapack yaml-cpp netcdf xerces-c make
```
**Windows:**
TO DO

**Note:** Be aware that some features might need to be enabled in the ESMF configuration to be available.


**Python dependencies:**

Ensure you have intake-esm, xarray, xesmf, gcsfs, swifter, xmip, cartopy and datatree installed.

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

    driver_vars.tex: Exports \newcommand macros with exact summary statistics (e.g., Taylor decomposition $R^2$ metrics and global driver dominance percentages) that update automatically upon parameter           changes.


## 📜 Citation

If you use this framework or the NPZ-T model in your research, please cite the following manuscript:

```bibtex
@misc{Neto2026NPZT,
  title={Emerging Thermal-Metabolic Control of Phytoplankton Annual Cycles: Extending the Critical Depth Hypothesis},
  author={Neto, Matías A. and Marquet, Pablo A. and Freilich, Mara A. and Martí, Luis and Sanchez-Pi, Nayat},
  year={2026},
  note={Submitted to SIAM Journal on Life Sciences (SIALS)},
  doi={10.48550/arXiv.2604.14759}
}

``

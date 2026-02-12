# JAX-CFD Extended: Subgrid-Scale (SGS) Modeling with JAX-CFD

**Authors:** Matteo Ugliotti, Mateo Reynoso

Actively maintained by Matteo Ugliotti. Please feel free to reach out to me (matteougliotti@gatech.edu) with any questions or cool project/implementation ideas.

---

## Overview

**JAX-CFD Extended** is an SGS-Modeling extension of the original [JAX-CFD](https://github.com/google/jax-cfd) framework developed by Google Research.  
It expands the numerical and modeling capabilities of JAX-CFD to support advanced turbulence research, interpretable data-driven closure modeling, and large-scale computational experiments. This project is an independent research extension and is not officially affiliated with Google Research.

### Who is this repository for?

This codebase is designed for:

- Researchers in computational fluid dynamics  
- Scientists studying turbulence and multiscale physics  
- Users interested in interpretable data-driven modeling  
- Graduate students building new LES closures

### What is this repository for?

The repository also includes implementations for reproducibility for the data used in my two papers, which can be found under notebooks/papers:

- **Data-driven modeling of multiscale phenomena with applications to fluid turbulence**  
  Brandon Choi, Matteo Ugliotti, Mateo Reynoso, Daniel R. Gurevich, Roman O. Grigoriev  
  Physical Review E (PRE), https://arxiv.org/abs/2511.09847, 2026

- **Physics-informed data-driven inference of an interpretable equivariant LES model of incompressible fluid turbulence**  
  Matteo Ugliotti, Brandon Choi, Mateo Reynoso, Daniel R. Gurevich, Roman O. Grigoriev
  Journal of Fluid Mechanics (JFM), (Arxiv Link coming soon) 2026

More details are provided in README's of each main subfolder.

Specifically, we focus on enabling modern research workflows in computational fluid dynamics:

- Large-eddy simulation (LES) with advanced subgrid-scale modeling in 2D
- Physics-informed and interpretable data-driven closures model discovery 
- Support for large DNS/LES datasets
- Tools for scalable numerical experiments  
- Early support for magnetohydrodynamics (MHD) and three-dimensional turbulence

Most of the changes we did to the original JAX-CFD are in the spectral subfolder. Specifically, we added two new files to handle SGS models (subgrid_models.py) and Chebychev grids (chebychev.py). We also added new integrators for MHD, 3D turbulence and compressible 2D turbulence (equations.py), new forcing functions (forcings.py) as well as lots of helper functions (utils.py). There are example scripts on how to use most our new features under notebooks with ample commenting.

### Compute Requirements

Small test runs can be executed on CPU, but meaningful DNS/LES experiments strongly benefit from GPU acceleration.

Large datasets may require substantial memory and storage.

### ⚡ Quick Start (Recommended)

If you want to run your first turbulence simulation as quickly as possible:

1. Install the package  
2. Open  
   👉 [Generate_IC.ipynb](Papers/Ugliotti_JFM_2026/Generate_Data/Generate_IC.ipynb)  
3. Then run  
   👉 [Generate_DNS.ipynb](Papers/Ugliotti_JFM_2026/Generate_Data/Generate_DNS.ipynb)

You will have a physically consistent DNS dataset in minutes.

From there you can:

- Run LES models  
- Perform diagnostics  
- Discover closures with SPIDER  

---

## Getting Started

This repository supports both **users who want to run simulations immediately** and **researchers aiming to extend the numerical framework**. The steps below outline the fastest path to a working setup while reflecting the scientific workflow used in the paper.

---

### 1. Install the package

Install directly from GitHub:

```bash
pip install git+https://github.com/cdggt/jax-cfd-sgs/
```

For development or modification, we recommend cloning instead:

```bash
git clone https://github.com/cdggt/jax-cfd-sgs.git
cd jax-cfd-sgs
pip install -e .
```

---

### 2. Install core dependencies

At minimum, ensure the following packages are available:

* `jax`
* `numpy`
* `xarray`
* `matplotlib`
* `tqdm`
* `pathlib`

If running on GPU/TPU, install the appropriate JAX build following the official guide:

👉 https://github.com/google/jax#installation

Install PySPIDER (required for model discovery workflows)

To reproduce the sparse-regression and equation-discovery components of the paper, install PySPIDER:

👉 https://github.com/sibirica/PySPIDER

Please follow the installation instructions provided in that repository to ensure all dependencies are configured correctly before running the SPIDER/ notebooks.

---

### 3. Run your first simulation (Recommended Scientific Order)

Example workflows are provided in:

```
Papers/Ugliotti_JFM_2026/Generate_Data/
```

We recommend following the pipeline below, which mirrors how turbulence datasets and models are constructed in practice:

1. [Generate_IC.ipynb](Papers/Ugliotti_JFM_2026/Generate_Data/Generate_IC.ipynb) — create a physically consistent initial condition
2. [Generate_DNS.ipynb](Papers/Ugliotti_JFM_2026/Generate_Data/Generate_DNS.ipynb) — run a high-fidelity DNS trajectory
3. [Generate_LES.ipynb](Papers/Ugliotti_JFM_2026/Generate_Data/Generate_LES.ipynb) — launch LES with supported SGS models
4. [Generate_NGMR.ipynb](Papers/Ugliotti_JFM_2026/Generate_Data/Generate_NGMR.ipynb) — run LES with the Reynolds-stress-based closure

These notebooks are heavily commented and serve as executable documentation of the framework.

---

### 4. Reproduce paper results

All reproducibility workflows are organized under:

```
Papers/Ugliotti_JFM_2026/
```

Follow the directories in **scientific workflow order**:

* **Generate_Data/** — produce DNS and LES datasets
* **SPIDER/** — perform sparse regression to discover NGMR from DNS
* **Analyze_Data/** — recreate figures and diagnostics

Each folder contains a dedicated README describing the pipeline.

---

### 5. Explore the spectral extensions

Most of the research functionality lives in:

```
upstream_jax_cfd/spectral/
```

Key additions include:

* Advanced SGS closures (NGM4, NGMR, SIM, DM, DSMAG)
* Compressible, incompressible, and MHD integrators
* ETDRK4 time-stepping
* Spectral utilities for filtering, projections, and LCR analysis

See the local README in that directory for a detailed overview.

---

### Suggested Workflow

If you are new to the repository:

👉 Start with **Generate_Data → SPIDER → Analyze_Data**

This reflects the full scientific pipeline:

**initial condition → DNS → model discovery → LES → diagnostics**

---

### Notes

* Large simulations may require substantial compute resources (GPU strongly recommended).
* Dataset quality is critical for regression workflows.
* Conventions should remain consistent when reproducing paper results.

---

This structure is intended to make the repository both **immediately usable** and **research-ready**, whether your goal is running turbulence simulations, benchmarking closures, or developing new models.

---

## Citation

If you use this repository in your research, please cite the following works:
```
@article{Ugliotti2026,
  doi = {},
  url = {},
  author = {Ugliotti, Matteo and Choi, Brandon and Reynoso, Mateo and Gurevich, Daniel and Grigoriev, Roman},
  title = {Physics-informed data-driven inference of an interpretable equivariant LES model of incompressible fluid turbulence},
  publisher = {},
  year = {2026},
  copyright = {}
}

@article{Dresdner2022-Spectral-ML,
  doi = {10.48550/ARXIV.2207.00556},
  url = {https://arxiv.org/abs/2207.00556},
  author = {Dresdner, Gideon and Kochkov, Dmitrii and Norgaard, Peter and Zepeda-Núñez, Leonardo and Smith, Jamie A. and Brenner, Michael P. and Hoyer, Stephan},
  title = {Learning to correct spectral methods for simulating turbulent flows},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
If you use pySPIDER please also cite:
```
@article{gurevich_learning_2024,  
	author = {Gurevich, Daniel R. and Golden, Matthew R. and Reinbold, Patrick A.K. and Grigoriev, Roman O.},  
	title = {Learning fluid physics from highly turbulent data using sparse physics-informed discovery of empirical relations ({SPIDER})},  
 	journal = {Journal of Fluid Mechanics},  
	volume = {996},  
	year = {2024},  
	pages = {A25}  
},
@phdthesis{gurevich_phd,
  author  = "Daniel Gurevich",
  title   = "Data-driven inference of
symmetry-equivariant models of
natural phenomena",
  school  = "Princeton University",
  year    = "2025"
}
```
Citations help support continued development and ensure proper academic attribution.

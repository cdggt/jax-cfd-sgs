# JAX-CFD Extended: Subgrid-Scale (SGS) Modeling with JAX-CFD

**Authors:** Matteo Ugliotti, Mateo Reynoso

Actively maintained by Matteo Ugliotti. Please feel free to reach out to me (matteougliotti@gatech.edu) with any questions or cool project/implementation ideas.

---

## Overview

**JAX-CFD Extended** is a research-oriented extension of the original [JAX-CFD](https://github.com/google/jax-cfd) framework developed by Google Research.  
It expands the numerical and modeling capabilities of JAX-CFD to support advanced turbulence research, interpretable data-driven closure modeling, and large-scale computational experiments.

Specifically, we focus on enabling modern research workflows in computational fluid dynamics:

- Large-eddy simulation (LES) with advanced subgrid-scale modeling in 2D
- Physics-informed and interpretable data-driven closures model discovery 
- Support for large DNS/LES datasets
- Tools for scalable numerical experiments  
- Early support for magnetohydrodynamics (MHD) and three-dimensional turbulence

Most of the changes we did to the original JAX-CFD are in the spectral subfolder. Specifically, we added two new files to handle SGS models (subgrid_models.py) and Chebychev grids (chebychev.py). We also added new integrators for MHD, 3D turbulence and compressible 2D turbulence (equations.py), new forcing functions (forcings.py) as well as lots of helper functions (utils.py). There are example scripts on how to use most our new features under notebooks with ample commenting.

The repository also includes implementations for reproducibility for the data used in my two papers, which can be found under notebooks/papers:

- **Data-driven modeling of multiscale phenomena with applications to fluid turbulence**  
  Brandon Choi, Matteo Ugliotti, Mateo Reynoso, Daniel R. Gurevich, Roman O. Grigoriev  
  Physical Review E (PRE), https://arxiv.org/abs/2511.09847, 2026

- **Physics-informed data-driven inference of an interpretable equivariant LES model of incompressible fluid turbulence**  
  Matteo Ugliotti, Brandon Choi, Mateo Reynoso, Daniel R. Gurevich, Roman O. Grigoriev
  Journal of Fluid Mechanics (JFM), 2026

More details are provided in README's of each main subfolder.

---

## Getting Started

To be updated soon 

---

## Installation

Install directly from the repository:

```bash
pip install git+https://github.com/cdggt/jax-cfd-sgs/
```

We also recommend making sure you have installed: numpy, jax, pathlib, xarray, matplotlib, tqdm.

---


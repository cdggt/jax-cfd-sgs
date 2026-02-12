# spectral — Modifications Overview

This README documents the primary additions and modifications made to the original **spectral JAX-CFD integrator**. The goal is to clearly identify what has been extended relative to the upstream implementation and where to find usage examples.

---

## `equations.py`

We added several new integrators to support a broader class of fluid and plasma simulations:

* **2D Compressible Navier–Stokes Integrator**
  Designed for the configuration described in:
  https://repository.gatech.edu/entities/publication/95e8faef-ff4d-4ba7-b18f-5802f4f304df

  **Inputs:** velocity ( u ) and scalar field ( \phi ).

* **3D Incompressible Navier–Stokes Integrator**
  Implemented in velocity form with a projection step to enforce incompressibility by removing the pressure component.

  **Input:** velocity ( u ).

* **2D Incompressible Magnetohydrodynamics (MHD) Solver**
  Solves the coupled MHD equations for fluid velocity and magnetic field evolution.

  **Inputs:** velocity ( u ) and magnetic field ( B ).

**Example usage:**
See the DNS workflows:

* [Generate_DNS.ipynb](/Papers/Ugliotti_JFM_2026/Generate_Data/Generate_DNS.ipynb)
* [Generate_IC.ipynb](/Papers/Ugliotti_JFM_2026/Generate_Data/Generate_IC.ipynb)

---

## `forcings.py`

Added multiple forcing mechanisms suitable for different turbulence regimes, including:

* Kolmogorov forcing
* Checkerboard forcing

These forcings support **2D**, **compressible**, and **3D** flow configurations.

---

## `time_stepping.py`

Added an **ETDRK4** time-stepping scheme (Exponential Time-Differencing Runge–Kutta 4, Krogstad variant).

**Why it is useful:**

* Provides improved stability for stiff PDEs.
* Allows larger timesteps compared to standard explicit methods.
* Particularly beneficial for **compressible flows**, where fast wave dynamics can otherwise impose restrictive CFL constraints.

---

## `subgrid_models.py`

Introduced a dedicated module implementing several subgrid-scale (SGS) closures:

* **NGM4** — Nonlinear Gradient Model (fourth order)
* **NGMR** — Reynolds-stress-based nonlinear gradient model
* **SIM** — Similarity model
* **DM** — Dynamic Mixed model
* **DSMAG** — Dynamic Smagorinsky

**Example usage:**

* [Generate_LES.ipynb](/Papers/Ugliotti_JFM_2026/Generate_Data/Generate_LES.ipynb)
* [Generate_NGMR.ipynb](/Papers/Ugliotti_JFM_2026/Generate_Data/Generate_NGMR.ipynb)

---

## `utils.py`

Extended with utilities frequently required in spectral turbulence workflows, including:

* Fourier filtering
* Projection to coarser grids
* Upsampling to finer grids
* Dealiasing
* Spectral derivatives
* Chebyshev-related utilities
* Leonard–Cross–Reynolds (LCR) decomposition
* Flux computations

For a complete description of available helpers, refer directly to the source file.

---

## Design Intent

These extensions aim to transform the original spectral integrator into a flexible research framework capable of supporting:

* DNS and LES workflows
* Compressible and incompressible dynamics
* MHD simulations
* Advanced SGS modeling

while maintaining compatibility with the upstream JAX-CFD architecture.

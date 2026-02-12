# How to analyze Data

## Overview

This folder contains the analysis scripts used to **quantitatively evaluate DNS and LES datasets** and reproduce the primary figures presented in the paper. The workflows focus on extracting physically meaningful diagnostics, assessing subgrid-scale (SGS) behavior, and benchmarking model performance across filter scales.

All scripts assume access to **DNS data** (and LES data where specified) stored in NetCDF format with the appropriate simulation metadata.

The directory includes a set of standalone analysis notebooks, each designed to reproduce a specific diagnostic or figure from the paper.

---

## Contents

### 1. Calculate Lengthscales ([Calculate-Lengthscale.ipynb](Calculate-Lengthscale.ipynb))

**Purpose:** Compute the characteristic system length scale required for subsequent analyses.

**Description:**

This script takes DNS velocity data as input and computes the length scale defined in the paper (see Eq. 2.9.)

ℓᵢ = sqrt( ⟨ || (1/12)(∇ₖuᵢ)(∇ₖuⱼ) ||_F ⟩_t  
      / ⟨ ∫ || (1/288)(∇ₖ∇ₗuᵢ)(∇ₖ∇ₗuⱼ) ||_F ⟩_t )

**Why this matters:**

The computed length scale is required as an input for several downstream scripts. In particular, it enables normalization of the filter scale (\(\Delta\)) and helps identify physically meaningful LES regimes.

---

### 2. Analyze Spectra and Vorticity ([Vorticity&Spectrum-Plots.ipynb](Vorticity&Spectrum-Plots.ipynb))

**Recreates:** **Figures 5, 6, 10, 18, 19**

**Purpose:** Visualize flow structure and spectral content to check that your simulations are good.

**Workflow:**

- Takes DNS data as input.
- Selects a user-defined snapshot.
- Produces:
  - Vorticity field visualizations  
  - Energy spectra  

These diagnostics are useful both for figure generation and for qualitative inspection of flow behavior.

---

### 3. Find the correct LES Regime ([Dissipation-Ratios.ipynb](Dissipation-Ratios.ipynb))

**Recreates:** **Figure 17**

**Purpose:** Identify appropriate LES operating regimes to run your LES on.

**Workflow:**

- Takes DNS data as input.
- Computes the filter scale normalized by the length scale

- Computes the viscous-to-SGS dissipation ratio

- Generates the regime map used to guide LES parameter selection.

**Physical interpretation:**

- **\(0 \ll \delta < 1\)**  
  Ensures the filter is large enough to justify LES but still small enough to accurately represent the SGS stress tensor.

- **\(\delta > 1\)**  
  The closure assumption breaks down because the unresolved stress cannot be faithfully represented.

- **\(Re_\delta > 1\) (preferably \(\gg 1\))**  
  Indicates that SGS dissipation dominates viscous dissipation — a necessary condition for testing model behavior at higher Reynolds numbers.

If \(Re_\delta\) is not sufficiently large, the experiment provides limited insight into model performance in turbulence-dominated regimes.

---

### 4. Plot fluxes ([Flux-Panel.ipynb](Flux-Panel.ipynb))

**Recreates:** **Figure 3**

**Purpose:** Compare energy flux predictions across SGS models (a-priori) to get intuition on how your runs might perform.

**Workflow:**

- Uses DNS data to compute the analytical SGS flux.
- Evaluates flux predictions from:
  - NGM4  
  - NGMR  
  - Similarity  
  - Dynamic Mixed (DM)  
  - Dynamic Smagorinsky (DSMAG)

The resulting panel provides a direct visual comparison between modeled and analytical flux behavior.

---

### 5. LCR accuracy ([LCR-Correlations.ipynb](LCR-Correlations.ipynb))

**Recreates:** **Figure 16**

**Purpose:** Analyze the Leonard–Cross–Reynolds (LCR) decomposition of the SGS tensor to find out what is important when. 

**Workflow:**

- Filters DNS data across multiple filter scales.
- Decomposes the SGS tensor into Leonard (L), Cross (C), and Reynolds (R) contributions.
- Computes three diagnostics:

1. Correlation of each tensor with the true SGS tensor \( \tau \)  
2. Correlation of each tensor’s flux with the true flux  
3. Dissipation contributed by each tensor relative to total SGS dissipation  

This analysis clarifies how each component contributes to the overall stress and energy transfer, indicating what needs to be modeled and why.

---

### 6. LES Models correlations ([Models-Correlations.ipynb](Models-Correlations.ipynb))

**Recreates:** **Figure 2**

**Purpose:** Benchmark SGS model accuracy against the analytical tensor.

**Workflow:**

- Computes the analytical SGS tensor from DNS.
- Evaluates modeled tensors from:
  - NGM4  
  - NGMR  
  - Similarity  
  - Dynamic Mixed  
  - Dynamic Smagorinsky  

across a range of filter scales.

The script then computes:

1. Tensor correlation with \( \tau \)  
2. Flux correlation with the analytical flux  
3. Relative dissipation compared to the true SGS dissipation  

Together, these metrics provide a comprehensive assessment of model fidelity.

---

### 7. R evolution equation accuracy([R-Correlations.ipynb](R-Correlations.ipynb))

**Recreates:** **Figure 4**

**Purpose:** Validate the evolution equation for the Reynolds stress tensor \(R\).

**Method:**

The script computes \( \partial_t R \) in two independent ways:

**(1) Using the evolution equation**

∂ₜRᵢⱼ = − ūₗ ∇ₗRᵢⱼ  + Rᵢₗ ∇ₗūⱼ  + Rⱼₗ ∇ₗūᵢ  + ν ∇²Rᵢⱼ  − S̄ₗₘ Rₗₘ δᵢⱼ  + k S̄ᵢⱼ  − α I Rᵢⱼ  + Dᵢⱼ⁽²⁾

**(2) Using a second-order Euler time discretization.**

The correlation between the resulting tensors quantifies the accuracy of the proposed evolution equation.

---

### 8. Vorticity Time-Correlation ([Vorticity-Correlation.ipynbs](Vorticity-Correlation.ipynbs))

**Recreates:** **Figure 9**

**Purpose:** Quantitatively evaluate how well LES trajectories track filtered DNS dynamics.

**Critical requirement:**  
DNS and LES runs must originate from the **same initial condition** and share identical simulation parameters (e.g., viscosity, forcing). Without this consistency, trajectory comparisons are not physically meaningful.

**Workflow:**

- Filters the DNS data at the LES resolution.
- Dealiases and downsamples the filtered DNS field to the LES grid, producing **FDNS**.
- Computes the time-dependent correlation between FDNS and LES trajectories.
- Generates correlation plots to assess predictive skill.

This provides a direct measure of how faithfully each LES model reproduces the resolved dynamics.

---

## Design Philosophy

The scripts in this folder emphasize:

- **Physical interpretability** — diagnostics are grounded in turbulence theory.  
- **Reproducibility** — figures can be regenerated directly from stored datasets.  
- **Comparability** — consistent metrics allow fair evaluation across models and filter regimes.  

Together, these tools form the quantitative backbone of the paper’s validation framework.

# SPIDER

## Overview

This folder contains **exploratory and learning scripts** demonstrating how sparse regression can be used to recover subgrid-scale (SGS) structure and evolution equations directly from DNS data using **PySPIDER**.  

The notebooks are intended to provide transparency into the model-discovery process presented in the paper and serve as reproducible examples for researchers interested in equation discovery for turbulence closures.

---

## Requirements

Before running the notebooks, install **PySPIDER** along with its dependencies:

👉 https://github.com/sibirica/PySPIDER  

Please follow the installation instructions provided in that repository.

Both scripts assume access to DNS datasets generated with:

👉 [Generate_DNS.ipynb](../Generate_Data/Generate_DNS.ipynb)

---

## Contents

### 1. Discovering NGM4 ([Finding-NGM4.ipynb](Finding-NGM4.ipynb))

**Purpose:** Recover the nonlinear gradient expansion directly from DNS data using sparse regression.

**Workflow:**

- Uses DNS vorticity data \( (t, x, y) \) as input.
- Constructs:
  - The analytical SGS tensor  
  - Filtered velocity field  
  - Filtered pressure field  

across a range of user-defined filter scales.

The notebook then performs a two-stage discovery process:

**Stage 1 — Recover the first-order correction**

- Runs SPIDER on the SGS tensor to identify the leading-order term.
- Recovers the nonlinear gradient model corresponding to **Eq. (3.1)** in the paper.

**Stage 2 — Recover the next-order correction**

- Subtracts the analytical NGM2 contribution from the SGS tensor.
- Runs SPIDER again on the residual.
- Identifies the higher-order correction corresponding to **Eq. (3.2)**.

Finally, the discovered coefficients are compared against the theoretical values predicted by the nonlinear gradient expansion.

**Outcome:**  
Demonstrates that sparse regression can recover the structure and coefficients of the NGM hierarchy directly from DNS data.

---

### 2. Discovering the Reynolds-Stress Evolution Equation ([Finding-R.ipynb](Finding-R.ipynb))

**Purpose:** Infer the evolution equation for the Reynolds stress tensor directly from DNS data.

**Workflow:**

- Uses DNS vorticity data to construct:
  - The Reynolds stress tensor  
  - Filtered velocity field  
  - Filtered pressure field  

- Computes the time derivative of the tensor.
- Applies SPIDER to perform sparse regression on the candidate operators.
- Analyzes the resulting regression terms to identify the governing dynamics.

**Outcome:**  
Recovers the Reynolds-stress evolution equation presented in **Eq. (3.4)** of the paper.

---

## Notes

- These notebooks are primarily intended for **interpretability and reproducibility**, rather than large-scale production workflows.
- DNS data quality is critical — noisy or under-resolved simulations may degrade regression performance.

---

## Design Philosophy

This folder emphasizes:

- **Scientific transparency** — exposing the equation-discovery process.  
- **Reproducibility** — enabling others to verify the recovered models.  
- **Interpretability** — illustrating how SGS structure emerges from data-driven regression.  

Together, these scripts document the methodological foundation behind the models introduced in the paper.

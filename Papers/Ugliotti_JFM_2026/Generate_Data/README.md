# How to generate Data

## Overview

This folder provides a **fully reproducible workflow** for generating both **Direct Numerical Simulation (DNS)** and **Large-Eddy Simulation (LES)** datasets used throughout the paper. The scripts are designed to clearly document the full pipeline. That is from constructing physically meaningful initial conditions to running high-fidelity simulations and LES and saving metadata-rich outputs for downstream analysis.

The directory contains **four example scripts** illustrating the primary data-generation workflows for single runs, as well as a `Batched_Runs` subfolder for large-scale production runs.

---

## Contents

### 1. Generate Initial Conditions ([Generate_IC.ipynb](Generate_IC.ipynb))
**Purpose:** Construct physically consistent initial conditions starting from a randomized field.

**Workflow:**
- Creates a random, divergence-free, non-physical vorticity field.
- Integrates the field forward in time using DNS until it reaches a statistically physical state (or not if you want to stop earlier you can)
- Includes optional diagnostics to verify the evolution:
  - Energy vs. time  
  - Energy spectrum  
  - Visualization of the final vorticity field
- Saves the final timestep for downstream simulations.

The resulting initial condition is **qualitatively similar to F4** in the paper.  
Both the vorticity field and all defining simulation attributes are saved in a **NetCDF (`.nc`) file**, ensuring full reproducibility.

---

### 2. Generate DNS Data from an Initial Condition ([Generate_DNS.ipynb](Generate_DNS.ipynb))
**Purpose:** Run DNS starting from a precomputed initial condition.

**Workflow:**
- Imports an initial condition.
- Accepts user-defined parameters such as:
  - Forcing configuration (including forcing type)
  - Viscosity
  - Additional runtime parameters
- Executes a DNS integration burst where the user specifies:
  - `TOTAL_TIME` — total integration duration  
  - `SAVE_EVERY` — snapshot frequency  

The time-resolved vorticity field, along with all simulation attributes, is saved in **NetCDF format**.

---

### 3. Generate LES Runs (NGM4, Similarity, Dynamic Smagorinsky, Dynamic Mixed) ([Generate_LES.ipynb](Generate_LES.ipynb))
**Purpose:** Demonstrate LES workflows using several widely adopted subgrid-scale models.

**Supported models:**
- Nonlinear Gradient Model to Fourth Order (**NGM4**)  
- **Similarity**  
- **Dynamic Smagorinsky**  
- **Dynamic Mixed**

**Workflow:**
- Imports an initial condition.
- Requests simulation parameters (forcing, viscosity, etc.).
- Filters and downsamples the initial condition to the desired LES resolution.
- Runs an LES integration burst with configurable:
  - `TOTAL_TIME`
  - `SAVE_EVERY`

Outputs include the full vorticity trajectory and all simulation metadata, saved to **NetCDF**.

---

### 4. Generate LES Runs for the NGMR Model ([Generate_NGMR.ipynb](Generate_NGMR.ipynb))
**Purpose:** Execute LES using the **NGMR** closure, which requires evolving an additional tensor field.

This script is separated from the other LES workflows because NGMR fundamentally differs in that it **explicitly evolves the Reynolds stress tensor**.

**Workflow:**
- Imports an initial condition and simulation parameters.
- Filters and downsamples the initial condition to the LES grid.
- Automatically computes the initial condition for the additional tensor field \( R \).
- Evolves both the vorticity field and the tensor during LES integration.
- Uses configurable `TOTAL_TIME` and `SAVE_EVERY`.

All outputs — including the tensor field and metadata — are stored in **NetCDF format**.

---

## Batched_Runs ([Batch_Runs](Batch_Runs))

The `Batched_Runs` subfolder contains scripts for executing **long-horizon forced DNS and LES simulations across multiple realizations simultaneously**.

These scripts were used to generate the forced data for **F4 and F5** in the paper.

**Key features:**
- Designed for extended integrations  
- Supports multiple realizations in parallel  
- Structured for large-scale dataset production  

Adapting these scripts for non-forced simulations or other models is straightforward and requires only switching off the focing.

---

## Design Principles

The scripts in this folder emphasize:

- **Reproducibility** — every dataset is saved with the attributes required to recreate the simulation.  
- **Transparency** — diagnostic tools help verify physical consistency.  
- **Modularity** — workflows can be easily adapted to new parameter regimes or models.  

Together, these examples provide a clear template for generating high-quality turbulence datasets suitable for benchmarking, model development, and scientific analysis.

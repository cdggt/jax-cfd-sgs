# Contributing to JAX-CFD Extended

Thank you for your interest in contributing! This repository is an actively developed research codebase focused on turbulence modeling, interpretable data-driven closures, and large-scale CFD workflows. Contributions that improve usability, performance, scientific clarity, and reproducibility are highly encouraged.

---

## Philosophy

This project prioritizes:

* **Scientific correctness**
* **Reproducibility**
* **Readable numerical implementations**
* **Interpretable modeling**
* **Well-documented workflows**

When contributing, please aim to maintain these standards.

---

## Ways to Contribute

We welcome several types of contributions:

### 🧠 New Physics / Numerical Methods

Examples include:

* New SGS closures
* Improved integrators
* Additional forcing mechanisms
* Extensions to MHD or compressible flows
* Higher-order numerical schemes

Please provide references when implementing methods derived from the literature.

---

### ⚡ Performance Improvements

* GPU efficiency
* Memory reductions
* Faster spectral operations
* JAX optimization

Include benchmarks whenever possible.

---

### 🐛 Bug Fixes

If you find a bug:

1. Open an issue describing the problem.
2. Provide a minimal reproducible example.
3. Include system details if relevant (CPU/GPU, JAX version, etc.).

---

### 📚 Documentation Improvements

Great documentation is critical for research adoption. Helpful contributions include:

* Clarifying README sections
* Improving notebook explanations
* Adding usage examples
* Fixing typos or ambiguous descriptions

---

## Before You Start

Please check existing issues and pull requests to avoid duplicate work.

For large features or architectural changes, open an issue first to discuss the proposal.

---

## Pull Request Guidelines

To help us review efficiently:

✅ Explain **what** you changed
✅ Explain **why** it matters
✅ Reference relevant papers if applicable

If your change affects numerical results, briefly describe the expected impact.

---

## Code Style

While we do not enforce a strict formatter, please follow these guidelines:

* Prefer clarity over cleverness.
* Avoid unnecessary abstraction in numerical code.
* Document non-obvious math.
* Use descriptive variable names when possible.
* Keep functions modular.

Comments explaining the physics are strongly encouraged.

---

## Testing Expectations

If applicable:

* Verify that existing workflows still run.
* Test notebooks impacted by your change.
* Confirm numerical behavior is reasonable.

For major numerical additions, a small validation example is highly appreciated.

---

## Notebook Contributions

When adding notebooks:

* Include explanatory markdown cells.
* Make the workflow reproducible.
* Avoid hidden preprocessing steps.
* Clearly state required inputs.

Think of notebooks as executable papers.

---

## Scientific Attribution

If your contribution implements or adapts a published method, please cite the original work in comments or documentation.

Academic transparency is a core value of this repository.

---

## Questions?

Feel free to open an issue for:

* Design discussions
* Feature ideas
* Clarifications
* Research collaboration

---

We appreciate your interest in improving this project and helping advance open, reproducible turbulence research.

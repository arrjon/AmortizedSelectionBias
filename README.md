# Overcoming Selection Bias in Statistical Studies With Amortized Bayesian Inference

**Jonas Arruda, Sophie Chervet, Paula Staudt, Andreas Wieser, Michael Hoelscher, Isabelle Sermet-Gaudelus, Nadine Binder, Lulla Opatowski, Jan Hasenauer**


---

## Abstract

Selection bias arises when the probability that an observation enters a dataset depends on variables related to the quantities of interest, leading to systematic distortions in estimation and uncertainty quantification. 
For example, in epidemiological or survey settings, individuals with certain outcomes may be more likely to be included, resulting in biased prevalence estimates with potentially substantial downstream impact.

Classical corrections, such as inverse-probability weighting or explicit likelihood-based models of the selection process, rely on tractable likelihoods, which limits their applicability in complex stochastic models with latent dynamics or high-dimensional structure.
Simulation-based inference enables Bayesian analysis without tractable likelihoods but typically assumes missingness at random and thus fails when selection depends on unobserved outcomes or covariates. 

Here, we develop a bias-aware simulation-based inference framework that explicitly incorporates selection into neural posterior estimation.
By embedding the selection mechanism directly into the generative simulator, the approach enables amortized Bayesian inference without requiring tractable likelihoods. 
This recasting of selection bias as part of the simulation process allows us to both obtain debiased estimates and explicitly test for the presence of bias. The framework integrates diagnostics to detect discrepancies between simulated and observed data and to assess posterior calibration.

The method recovers well-calibrated posterior distributions across three statistical applications with diverse selection mechanisms, including settings in which likelihood-based approaches yield biased estimates. These results recast the correction of selection bias as a simulation problem and establish simulation-based inference as a practical and testable strategy for parameter estimation under selection bias.

---

## Repository Structure

```
AmortizedSelectionBias/
├── KoCo/                    # Prevalence estimation (KoCo19 Study)
├── visit_censoring/         # Dementia progression (Framingham Heart Study)
└── PedCov/                  # COVID-19 child-depended inclusion (PedCovid Study)
```

### Examples

| Directory | Description |
|---|---|
| `KoCo/` | Prevalence estimation under outcome-dependent missingness and selection. Demonstrates bias on simulated data and applies the framework to real seroprevalence data. |
| `visit_censoring/` | Longitudinal illness-death model with visit-censoring. Validates debiasing via simulation-based calibration (SBC) and compares against naive, and spline-based approaches. |
| `PedCov/` | Real-world example with a complex, multi-mechanism selection process. Trains a single amortized model across selection scenarios and validates against MCMC. |

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Python 3.11 or later is required.

```bash
# Install uv (if not already installed)
curl -Lsf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

Some examples additionally require R and R packages (Stan etc.).
They can be installed via the `install_requirements.py` script:

```bash
uv run python install_requirements.py
```
---

## Dependencies

- [BayesFlow](https://bayesflow.org/) — amortized Bayesian inference via neural posterior estimation
- [JAX](https://github.com/google/jax) — deep learning backend
- [CmdStanPy](https://github.com/stan-dev/cmdstanpy) — Stan interface for likelihood-based comparisons
- [rpy2](https://rpy2.github.io) — R to Python interface

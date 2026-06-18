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
Install uv and requirements with:

```bash
# Install uv (if not already installed)
curl -Lsf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```
This should take not more than a few minutes.
Some examples additionally require R and R packages (Stan etc.).
They can be installed via the `install_requirements.py` script:

```bash
uv run python install_requirements.py
```

To train on a GPU, use
```bash
uv pip install --upgrade "jax[cuda13]"
```
and ensure that the required CUDA drivers and libraries are installed on your system (see [JAX GPU support](https://docs.jax.dev/en/latest/installation.html)).

---

## Dependencies

- [BayesFlow](https://bayesflow.org/) — amortized Bayesian inference via neural posterior estimation
- [JAX](https://github.com/google/jax) — deep learning backend
- [CmdStanPy](https://github.com/stan-dev/cmdstanpy) — Stan interface for likelihood-based comparisons
- [rpy2](https://rpy2.github.io) — R to Python interface

---

## Instructions to run the examples

Each example is a self-contained workflow split into a simulator module (`*_simulate.py`, defines the prior and the bias-aware generative model) and an inference module (`*_inference.py`, builds the neural network, runs amortized inference, and produces the diagnostics and figures). 

All networks are saved in each `models/` directory, so after training once inference runs without retraining.


Run the example modules from the repository root:

```bash
# Prevalence estimation (KoCo19)
uv run python -m KoCo.prevalence_inference

# Dementia progression / visit-censoring (Framingham)
uv run python -m visit_censoring.cens_visit_inference

# Multi-mechanism selection (PedCovid)
uv run python -m PedCov.pedcov_inference
```

By default, the inference scripts load the pretrained network from `models/` if available and run on the bundled data in the corresponding `data/` directory. 
Training from scratch is triggered when no matching model file is present.

### Expected output

- **Posterior samples** for the parameters of interest, debiased for the selection mechanism.
- **Diagnostic figures** written to the example's `plots/` directory (PDF), including simulation-based calibration (SBC) plots, posterior recovery / contraction, and classifier two-sample test (C2ST) checks comparing simulated and observed data to detect residual bias.

### Expected run time for demo on a "normal" desktop computer

Using the pretrained networks, each demo completes in roughly **2–10 minutes** on a typical desktop CPU (most of the time is spent generating posterior samples and rendering the diagnostic figures). 
Training a network from scratch is substantially more demanding: simulated training data has to be generated (depending on the number of CPU can take a couple of hours) and neural networks have to be trained (couple of hours on a GPU).

---


## How to run the method on your data

The core requirement for applying the framework to your own study is to **define a simulator** for your problem: a generative model that matches your data-generating process and that embeds your selection/inclusion mechanism directly into the simulation (this is what makes the inference bias-aware), together with a prior over the parameters of interest. 
The `*_simulate.py` modules in this repository serve as worked templates.

Once you have a simulator, the rest of the workflow is the standard amortized Bayesian inference pipeline — building a neural approximator, training it on simulations, and applying it to your observed data.
Follow the [BayesFlow documentation](https://bayesflow.org/) for the general workflow and API; the `*_inference.py` modules here show how each example wires its simulator into that pipeline and adds the SBC and C2ST diagnostics used to confirm posterior calibration and test for residual selection bias.

The `KoCo/` example is the simplest starting point for a new prevalence-style application; `PedCov/` shows how to amortize a single network across multiple selection scenarios.
The `visit_censoring/` simulator cannot run without access to data, as covariates are directly embedded into the generative model.

## Reproduction instructions

To reproduce the paper's results from scratch one would need access to the data. Here we can only provide instructions for the synthetic data examples:

1. Install all dependencies, including the R/Stan toolchain used for the likelihood-based comparisons:
   ```bash
   uv sync
   uv run python install_requirements.py
   ```
2. Run each `*_inference.py` module as shown above. The training scripts honour the `SLURM_ARRAY_TASK_ID` and `SLURM_CPUS_PER_TASK` environment variables for parallel execution on a cluster.
3. The resulting figures are written to each example's `plots/` directory and correspond to those reported in the manuscript.

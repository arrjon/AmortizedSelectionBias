"""Bayesian MRP baseline: logistic outcome model + poststratification onto the raked sample."""
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from KoCo.prevalence_simulate import rake, test_specificity_params

BASE = Path(__file__).resolve().parent
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
stan_model = CmdStanModel(stan_file=BASE / 'mrp_stan_model.stan')


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """N x 10 dummies in make_priors() coefficient order."""
    return np.column_stack([
        df['sex'] == 'f',
        df['age_group'] == '0-19',
        df['age_group'] == '35-49',
        df['age_group'] == '50-64',
        df['age_group'] == '65-79',
        df['age_group'] == '80+',
        df['birth_country'] == 1,
        df['hh_size'] == 'household_2',
        df['hh_size'] == 'household_34',
        df['hh_size'] == 'household_5+',
    ]).astype(float)


def poststrat_cells(sample_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Poststratification cells: MRP rakes the observed sample to the Munich marginals itself."""
    df = sample_df.assign(_w=rake(sample_df))
    cells = (
        df
        .groupby(['age_group', 'sex', 'hh_size', 'birth_country'], dropna=False, observed=True)['_w']
        .sum()
        .reset_index()
    )
    return design_matrix(cells), cells['_w'].values


def fit_mrp(
    sample_df: pd.DataFrame,
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Posterior draws of 'prevalence' (poststratified), 'alpha' and 'beta' (N_draws x 10, in
    make_priors() order). Likelihood uses complete cases only."""
    obs = sample_df.dropna(subset=['y_test'])
    x_cells, n_c = poststrat_cells(sample_df)
    with tempfile.TemporaryDirectory() as out_dir:  # cmdstanpy otherwise leaks a temp dir per fit
        fit = stan_model.sample(
            data={
                'N': len(obs),
                'y': obs['y_test'].astype(int).tolist(),
                'X': design_matrix(obs),
                'C': len(n_c),
                'X_cells': x_cells,
                'N_c': n_c,
                'sens': test_specificity_params['sensitivity'],
                'spec': test_specificity_params['specificity'],
            },
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            output_dir=out_dir,
            show_progress=False,
            show_console=False,
        )
        return {v: fit.stan_variable(v) for v in ('prevalence', 'alpha', 'beta')}


if __name__ == '__main__':
    from KoCo.prevalence_simulate import simulate_population, full_population_size

    sim_out = simulate_population(
        epoch_index=1, n_out=int(full_population_size * 0.1), with_missingness=False, seed=0
    )
    t0 = time.perf_counter()
    draws = fit_mrp(sim_out['subsample'], seed=0)['prevalence']
    dt = time.perf_counter() - t0

    prev_true = sim_out['prevalence_true']
    median = float(np.median(draws))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f'fit time: {dt:.1f} s (4 chains x 500 draws)')
    print(f'prevalence_true: {prev_true * 100:.3f}%')
    print(f'MRP median:      {median * 100:.3f}%  95% CI [{lo * 100:.3f}, {hi * 100:.3f}]')
    assert abs(median - prev_true) < 0.01, f'median off by {(median - prev_true) * 100:.3f} pp'
    assert lo <= prev_true <= hi, '95% CI does not contain prevalence_true'
    print('self-check passed')

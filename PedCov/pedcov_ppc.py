"""Posterior predictive checks for the PedCov example.

Pushes posterior draws from both methods through the generative model each method
assumed and compares summary statistics of the re-simulated studies to the
observed ones.

    uv run python -m PedCov.pedcov_ppc
"""
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde

BASE = Path(__file__).resolve().parent
networks = ['flow_matching_tt_300']  # bias-aware NPE networks to check
variants = ['alpha', 'omicron']
alpha_fixed = {'alpha': 0.001, 'omicron': 0.01}  # see prior() in pedcov_inference.py
param_names = ['beta', 'delta',
               'mu_inf_SI', 'mu_inf_SC', 'mu_inf_AI', 'mu_inf_AC', 'mu_inf_AA',
               'mu_susc_I', 'mu_susc_C', 'mu_protect_acq', 'mu_protect_transm']
# Recruitment mechanism each method's replicates are generated under. Both are assumptions: for
# the real study the mechanism is not observed, we only assume it matches the PedCov child-based
# recruitment, while MCMC re-simulates under the random recruitment its own model assumes.
method_selection = {'npe': 'pedcov', 'mcmc': 'random'}
method_labels = {'npe': 'Bias-aware NPE', 'mcmc': 'MCMC (random recruitment)'}
method_colors = {'npe': '#1B8A8F', 'mcmc': '#4B2E83'}

n_draws = 100
n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
seed = 0
cache_file = BASE / 'models' / 'ppc_sims.pkl'  # {(variant, 'npe:<network>' | 'mcmc'): [sim_data]}

# feature layout of normalize_household_data (see helper_functions.process_household)
F_DATE_SYMPT, F_STATUS_SYM, F_STATUS_ASYM, F_AGE = 0, 4, 5, 6
AGE_INFANT, AGE_CHILD, AGE_ADULT = 0, 1, 2

stat_labels = {
    # overall transmission (driven by beta)
    'sar': 'Secondary\nattack rate',
    'mean_infected_hh': 'Mean infected\nper household',
    'frac_hh_secondary': 'Households with\n≥1 secondary case',
    # infectiousness by age of the index case (mu_inf)
    'sar_index_infant': 'Secondary attack rate\ninfant index',
    'sar_index_child': 'Secondary attack rate\nchild index',
    'sar_index_adult': 'Secondary attack rate\nadult index',
}

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def _rate(num, den):
    return num / den if den > 0 else np.nan


def summary_stats(sim_data: np.ndarray) -> dict:
    """Summary statistics of one study in the normalized household layout.

    ``sim_data`` has shape (n_households, n_members, 11); all-zero rows/households are
    padding. Index case = earliest ``date_sympt`` among the infected household members
    (as in stan.py / count_households_first_pos); asymptomatic cases carry their detection
    date there, only uninfected members have -1. Secondary attack rates are computed among
    non-index members, so studies of different size stay comparable, and are split by the
    age group of the index case: that is the infectiousness side of the model (``mu_inf``),
    unlike a split by the age of the contact, which probes susceptibility.
    """
    arr = np.asarray(sim_data).reshape(-1, sim_data.shape[-2], sim_data.shape[-1])
    n_hh = n_hh_secondary = n_infected = 0
    by_index_age = {a: [0, 0] for a in (AGE_INFANT, AGE_CHILD, AGE_ADULT)}  # [secondary cases, contacts]
    for hh in arr:
        members = hh[np.abs(hh).sum(-1) > 1e-8]  # drop padded members
        infected = members[:, F_STATUS_SYM] + members[:, F_STATUS_ASYM] > 0.5
        if not infected.any():  # padded household, or nobody infected -> no index case
            continue
        n_hh += 1
        n_infected += infected.sum()
        index = np.flatnonzero(infected)[np.argmin(members[infected, F_DATE_SYMPT])]
        keep = np.ones(len(members), bool)
        keep[index] = False
        n_hh_secondary += infected[keep].any()
        group = by_index_age[int(np.round(members[index, F_AGE]))]
        group[0] += infected[keep].sum()
        group[1] += keep.sum()
    return {
        'sar': _rate(sum(g[0] for g in by_index_age.values()), sum(g[1] for g in by_index_age.values())),
        'mean_infected_hh': _rate(n_infected, n_hh),
        'frac_hh_secondary': _rate(n_hh_secondary, n_hh),
        'sar_index_infant': _rate(*by_index_age[AGE_INFANT]),
        'sar_index_child': _rate(*by_index_age[AGE_CHILD]),
        'sar_index_adult': _rate(*by_index_age[AGE_ADULT]),
    }


def simulate_study(variant: str, params: dict, selection: str) -> np.ndarray:
    # imported inside the worker: PedCov.simulator sources Simulator.R at import time
    from PedCov.simulator import OutbreakSimulator
    return OutbreakSimulator(variant)(**params, selection_procedure=selection, return_df=False)['sim_data']


def draw_params(samples: dict, variant: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    n_total = np.asarray(samples[param_names[0]]).size
    # prefix of a fixed permutation: raising n_draws keeps the draws already simulated
    idx = rng.permutation(n_total)[:min(n_draws, n_total)]
    return [{'alpha': alpha_fixed[variant],
             **{p: float(np.asarray(samples[p]).flatten()[i]) for p in param_names}}
            for i in idx]


def ppc_table(obs: dict, stats: dict, network: str) -> pd.DataFrame:
    rows = []
    for stat, label in stat_labels.items():
        for variant in variants:
            for method in method_labels:
                s = np.array([d[stat] for d in stats[(variant, method)]])
                s = s[~np.isnan(s)]
                o = obs[variant][stat]
                p = np.mean(s >= o)
                rows.append({
                    'network': network, 'statistic': label, 'variant': variant,
                    'method': method_labels[method], 'observed': o, 'median': np.median(s),
                    'q2.5': np.quantile(s, 0.025), 'q97.5': np.quantile(s, 0.975),
                    'p_value': min(p, 1 - p) * 2,
                })
    return pd.DataFrame(rows)


def plot_ppc(obs: dict, stats: dict, path: Path) -> None:
    fig, axes = plt.subplots(len(stat_labels), len(variants), figsize=(7, 1.5 * len(stat_labels)),
                             layout='constrained', sharex='row')
    for i, (stat, label) in enumerate(stat_labels.items()):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            values = {m: np.array([d[stat] for d in stats[(variant, m)]]) for m in method_labels}
            values = {m: v[~np.isnan(v)] for m, v in values.items()}
            pooled = np.concatenate(list(values.values()) + [[obs[variant][stat]]])
            pad = 0.05 * (pooled.max() - pooled.min())
            grid = np.linspace(pooled.min() - pad, pooled.max() + pad, 200)
            for method, method_label in method_labels.items():
                legend_label = method_label if i == 0 and j == 0 else None
                if values[method].std() == 0:  # identical draws, the kde is singular
                    ax.axvline(values[method][0], color=method_colors[method], alpha=0.55,
                               label=legend_label)
                    continue
                ax.fill_between(grid, gaussian_kde(values[method])(grid), color=method_colors[method],
                                alpha=0.55, lw=0, label=legend_label)
            ax.axvline(obs[variant][stat], color='black', lw=1.5,
                       label='Observed' if i == 0 and j == 0 else None)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(bottom=0)  # density axis starts at 0, the kdes sit on the x-axis
            if i == 0:
                ax.set_title(variant.title(), fontsize=16)
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.legend(loc='outside lower center', ncol=len(method_labels) + 1, frameon=False, fontsize=12)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def simulate_cached(sims: dict, key: tuple, variant: str, params: list[dict], selection: str) -> None:
    done = sims.get(key, [])
    if len(done) >= len(params):
        return
    todo = params[len(done):]  # only the draws not simulated yet
    t0 = time.time()
    try:
        studies = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(simulate_study)(variant, p, selection) for p in todo)
    except Exception as e:  # noqa: BLE001 - rpy2 in workers can be fragile
        print(f'parallel simulation failed ({e}), falling back to sequential')
        studies = [simulate_study(variant, p, selection) for p in todo]
    print(f'{key}: +{len(studies)} studies ({selection}) in {time.time() - t0:.0f}s, '
          f'n_jobs={n_jobs}, total={len(done) + len(studies)}')
    sims[key] = done + [np.asarray(s, dtype=np.float32) for s in studies]
    cache_file.write_bytes(pickle.dumps(sims))  # so a rerun only recomputes statistics


def main() -> None:
    # the observed study is only stored in normalized form, see pedcov_inference.py
    obs = {v: summary_stats(np.load(BASE / 'data' / f'{v}_sim_data.npy')) for v in variants}
    sims = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}

    # Stan posteriors are network independent (same pickle content for every network)
    for variant in variants:
        with open(BASE / 'models' / f'{variant}_{networks[0]}_npe_posterior_samples.pkl', 'rb') as f:
            stan = pickle.load(f)[variant]['stan_posterior_samples']
        simulate_cached(sims, (variant, 'mcmc'), variant, draw_params(stan, variant),
                        method_selection['mcmc'])

    tables = []
    for network in networks:
        for variant in variants:
            with open(BASE / 'models' / f'{variant}_{network}_npe_posterior_samples.pkl', 'rb') as f:
                npe = pickle.load(f)[variant]['posterior_samples']
            simulate_cached(sims, (variant, f'npe:{network}'), variant, draw_params(npe, variant),
                            method_selection['npe'])
        stats = {(v, 'mcmc'): [summary_stats(s) for s in sims[(v, 'mcmc')]] for v in variants}
        stats |= {(v, 'npe'): [summary_stats(s) for s in sims[(v, f'npe:{network}')]] for v in variants}
        table = ppc_table(obs, stats, network)
        table.to_csv(BASE / 'plots' / f'{network}_ppc.csv', index=False)
        print(f'\n=== {network} ===')
        print(table.drop(columns='network').to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        plot_ppc(obs, stats, BASE / 'plots' / f'{network}_ppc.pdf')
        tables.append(table)


if __name__ == '__main__':
    main()

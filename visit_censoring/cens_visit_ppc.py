"""Posterior predictive checks for the Framingham visit-censoring example.

Re-simulates the study from posterior draws each through the observation process that method assumed
(CensVisit vs. Full, the latter capped at 5 years as in training). As a third method the
frequentist spline illness-death model of Binder et al. (2019) enters as a plug-in
predictive, where estimation uncertainty is propagated by perturbing each baseline
cumulative hazard per replicate with the published confidence band.

Run from the repo root:  uv run python -m visit_censoring.cens_visit_ppc
"""
import json
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from visit_censoring.cens_visit_helper import compute_gamma_params
from visit_censoring.cens_visit_simulate import simulate_params

truncate_data = ro.globalenv['truncateData']

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path('/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/visit_censoring')
network_name = 'ConsistencyModel_SetTransformer'
n_draws = 200
ncores = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
epochs = ['epoch1', 'epoch2', 'epoch3', 'epoch4']

methods = {  # label -> (posterior suffix or None for the plug-in spline IDM, color, scheme)
    'Bias-aware NPE': ('', '#1B8A8F', 'CensVisit'),
    'NPE (no selection model)': ('_full', '#E68600', 'Full'),
    'Splines IDM': (None, '#D64A62', 'CensVisit'),
}
MAX_TIME = 1825  # Full-scheme data are capped at 5 years in cens_visit_inference.py

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# same priors as cens_visit_inference.py; only used for the epoch names on the R side
_shape = compute_gamma_params(1.0, cv=0.25)
_a = compute_gamma_params(0.0003, cv=1.0)
priors = {
    'a01': _a, 'a02': _a, 'a12': _a,
    'shape01': _shape, 'shape02': _shape, 'shape12': _shape,
    **{f'beta{t}_{c}': {'mean': 0.0, 'sd': 1.0}
       for t in ('01', '02', '12') for c in ('sex', 'age')},
}

stat_names = {
    'n_dementia': 'Dementia diagnoses',
    'n_death': 'Deaths',
    'n_death_no_dementia': 'Deaths without diagnosis',
    'median_death_time': 'Median death time [days]',
}


def summarize(df):
    """Four summary statistics of one illness-death dataset."""
    return {
        'n_dementia': float((df['ills'] == 1).sum()),
        'n_death': float((df['ds'] == 1).sum()),
        'n_death_no_dementia': float(((df['ds'] == 1) & (df['ills'] == 0)).sum()),
        'median_death_time': float(df.loc[df['ds'] == 1, 'dt'].median()),
    }


def spline_cumhaz(baseline_epoch, trans):
    """Baseline cumulative hazard of the spline IDM on a day grid."""
    d = baseline_epoch[f'H{trans}']['idm_splines']
    t = np.concatenate([[0.0], d['times'], [MAX_TIME]])
    h = np.array([d['cumhaz'][0], *d['cumhaz'], d['cumhaz'][-1]])
    return t, np.concatenate([[0.0], np.cumsum(np.diff(t) * (h[1:] + h[:-1]) / 2)])


def spline_cumhaz_sd(baseline_epoch, trans):
    """Lognormal sd of the spline IDM baseline cumulative hazard, from its confidence band."""
    d = baseline_epoch[f'H{trans}']['spl']
    lo, hi = np.array(d['ci_low']), np.array(d['ci_high'])
    ok = lo > 0
    return float(np.median(np.log(hi[ok] / lo[ok]) / (2 * 1.96)))


def simulate_spline_idm(epoch, n_rep, rng):
    """Plug-in replicates of the spline IDM at its point estimate."""
    df = pd.read_csv(BASE / 'data' / f'{epoch}_CV.csv')
    sex = df['sex'].to_numpy(float)
    age = df['age'].to_numpy(float)
    age = age - age.mean()  # as scale(cov_age, scale=FALSE) in R
    n = len(df)

    baseline = json.loads((BASE / 'baseline.json').read_text())[epoch]
    coef = {c['row']: c['idm_splines'] for c in baseline['coefs']}
    H0 = {tr: spline_cumhaz(baseline, tr) for tr in ('01', '02', '12')}
    sd = {tr: spline_cumhaz_sd(baseline, tr) for tr in H0}
    eta = {tr: coef[f'sex_{tr}'] * sex + coef[f'ageCentered_{tr}'] * age for tr in H0}

    def draw(trans, idx, offset):
        t, H = H0[trans]
        target = -np.log(rng.random(idx.size)) / np.exp(eta[trans][idx] + offset[trans])
        # target beyond H0(MAX_TIME) -> no event within follow-up (inf)
        return np.maximum(np.interp(target, H, t, right=np.inf), 1.0)

    # visit times exactly as in simulate_from_priors_df()
    visit1_typical = df.loc[(df['illt'] > 0) & (df['illt'] < 1000), 'illt'].median()
    visit2_typical = df.loc[(df['illt'] > 1000) & (df['illt'] < MAX_TIME), 'illt'].median()

    ids = np.arange(n)
    stats = []
    for _ in range(n_rep):
        # estimation uncertainty: one baseline-hazard factor per transition per replicate
        offset = {tr: sd[tr] * rng.normal() for tr in H0}
        t01, t02 = draw('01', ids, offset), draw('02', ids, offset)
        ills = (t01 <= t02).astype(int)
        illt = np.minimum(t01, t02)
        dt = illt.copy()
        ill = np.flatnonzero(ills)
        dt[ill] = illt[ill] + draw('12', ill, offset)
        full = pd.DataFrame({'id': ids + 1, 'illt': illt, 'ills': ills,
                             'dt': dt, 'ds': 1, 'sex': sex, 'age': age})
        v = np.sort(np.stack([visit1_typical + rng.normal(0, 100, n),
                              visit2_typical + rng.normal(0, 150, n)]), axis=0)
        v = np.clip(v, 0, MAX_TIME)
        with (ro.default_converter + pandas2ri.converter).context():
            cens = truncate_data(full, pd.DataFrame({'visit1': v[0], 'visit2': v[1]}),
                                 'illt', 'dt', MAX_TIME).getbyname('CensVisit')
        stats.append(summarize(cens.round({'illt': 0, 'dt': 0})))
    return stats


def simulate_stats(labels):
    """Predictive summary statistics: {label: {epoch: {stat: array(n_draws)}}}."""
    out = {}
    for label in labels:
        suffix, _, scheme = methods[label]
        if suffix is None:  # plug-in: no posterior, replicate at the point estimate
            rng = np.random.default_rng(0)
            out[label] = {}
            for epoch in epochs:
                t0 = time.time()
                stats = simulate_spline_idm(epoch, n_draws, rng)
                out[label][epoch] = {s: np.array([d[s] for d in stats]) for s in stat_names}
                print(f'{label} {epoch}: {n_draws} replicates in {time.time() - t0:.0f}s')
            continue
        with open(BASE / 'models' / f'posterior_samples_{network_name}{suffix}.pkl', 'rb') as f:
            post = pickle.load(f)  # {param: (n_epochs, n_samples, 1)}
        out[label] = {}
        for i, epoch in enumerate(epochs):
            draws = {k: v[i, :n_draws, 0][None, :] for k, v in post.items()}  # (1, n_draws)
            t0 = time.time()
            sim = simulate_params(draws, priors=priors, ncores=ncores, epochs=[epoch])
            sim = sim[sim['scheme'] == scheme].copy()
            if scheme == 'Full':  # same administrative cap as the NPE-full training data
                sim.loc[sim['dt'] > MAX_TIME, 'ds'] = 0
                sim.loc[sim['dt'] > MAX_TIME, 'dt'] = MAX_TIME
                sim.loc[sim['illt'] > MAX_TIME, 'ills'] = 0
                sim.loc[sim['illt'] > MAX_TIME, 'illt'] = MAX_TIME
            stats = [summarize(g) for _, g in sim.groupby('replicate', sort=True)]
            out[label][epoch] = {s: np.array([d[s] for d in stats]) for s in stat_names}
            print(f'{label} {epoch}: {n_draws} draws in {time.time() - t0:.0f}s')
    return out


def ppc_pvalue(sim, obs):
    p = float(np.mean(sim >= obs))
    return min(p, 1 - p) * 2


def main():
    observed = {e: summarize(pd.read_csv(BASE / 'data' / f'{e}_CV.csv')) for e in epochs}

    cache = BASE / 'models' / f'{network_name}_ppc_stats.pkl'
    sim_stats = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    missing = [m for m in methods if m not in sim_stats]
    if missing:
        print(f'Simulating {missing}, rest from cache {cache}')
        sim_stats.update(simulate_stats(missing))
        cache.write_bytes(pickle.dumps(sim_stats))
    else:
        print(f'Loaded cached predictive statistics from {cache}')

    rows = []
    for stat in stat_names:
        for e in epochs:
            for label in methods:
                s = sim_stats[label][e][stat]
                rows.append({
                    'statistic': stat_names[stat], 'epoch': e, 'method': label,
                    'observed': observed[e][stat],
                    'pred_median': np.median(s),
                    'pred_2.5%': np.quantile(s, 0.025),
                    'pred_97.5%': np.quantile(s, 0.975),
                    'ppp_value': ppc_pvalue(s, observed[e][stat]),
                })
    table = pd.DataFrame(rows)
    table.to_csv(BASE / 'plots' / f'{network_name}_ppc.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', None,
                           'display.float_format', '{:.3g}'.format):
        print(table.to_string(index=False))

    fig, axes = plt.subplots(len(stat_names), len(epochs), figsize=(11, 8.5), sharex='row', layout='constrained')
    for r, stat in enumerate(stat_names):
        for c, e in enumerate(epochs):
            ax = axes[r, c]
            all_vals = np.concatenate([sim_stats[m][e][stat] for m in methods] + [[observed[e][stat]]])
            bins = np.linspace(all_vals.min(), all_vals.max(), 25)
            for label, (_, color, _scheme) in methods.items():
                ax.hist(sim_stats[label][e][stat], bins=bins, color=color, alpha=0.55,
                        density=True, label=label)
            ax.axvline(observed[e][stat], color='black', lw=1.5, label='Observed')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=10)
            if r == 0:
                ax.set_title(f'Epoch {c + 1}', fontsize=14)
            if c == 0:
                ax.set_ylabel(stat_names[stat], fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False, fontsize=13)
    fig.savefig(BASE / 'plots' / f'{network_name}_ppc.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {BASE / 'plots' / f'{network_name}_ppc.pdf'}")

#%%
if __name__ == '__main__':
    main()

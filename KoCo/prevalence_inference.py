#%% uv pip install bayesflow==2.0.8
import os
os.environ['KERAS_BACKEND'] = 'jax'

from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pickle
from joblib import Parallel, delayed
import time

import keras
import bayesflow as bf

from KoCo.prevalence_simulate import simulate_population, full_population_size
from KoCo.mrp_baseline import fit_mrp

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path('/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/KoCo')
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))

cat_to_int = {
    'age_group': {'0-19': 0, '20-34': 1, '35-49': 2, '50-64': 3, '65-79': 4, '80+': 5},
    'sex': {'m': 0, 'f': 1},
    'hh_size': {'household_1': 1, 'household_2': 2, 'household_34': 3, 'household_5+': 4},
    'birth_country': {0: 0, 1: 1},
    'timepoint': {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T5': 5}
}
colors = ['#4B2E83', '#D64A62', '#1B8A8F']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def hyperparameters():
    return dict(epoch_index=np.random.choice([1,2,3,4,5]))


def convert_to_nn_input(new_population: pd.DataFrame, fill_na_value=-1) -> np.ndarray:
    """convert df to format for NN input"""
    new_population_nn = new_population.copy()
    # convert categorical variables to integers
    for cat in cat_to_int:
        new_population_nn[cat] = new_population_nn[cat].map(cat_to_int[cat])

    # keep only the relevant columns
    new_population_nn = new_population_nn[['y_test']+list(cat_to_int.keys())]

    # fill NA values in categorical variables
    for c in new_population_nn.columns:
        new_population_nn[c] = new_population_nn[c].fillna(fill_na_value)
    return new_population_nn.values


def simulator(epoch_index):
    out = simulate_population(
        epoch_index=epoch_index,
        n_out=int(full_population_size * 0.1)
    )
    # make data suitable for neural network
    data_df = out['subsample']
    data = convert_to_nn_input(data_df)

    return dict(
        prevalence_true=out['prevalence_true'],
        prevalence_subsample=out['prevalence_subsample'],
        prevalence_subsample_weighted=out['prevalence_subsample_weighted'],
        data=data
    )

simulator_bf = bf.make_simulator([hyperparameters, simulator])
#%%

data_path = BASE / 'data'
n_val_data = 1000
n_train_data = 10000
create_data = False
training_data, validation_data = None, None

if create_data:
    # Simulate and extract structured arrays
    logging.info('Generate validation data...')
    validation_data = simulator_bf.sample_parallel(n_val_data)
    with open(data_path / 'validation_data.pkl', 'wb') as f:
       pickle.dump(validation_data, f)

    logging.info('Generate training data...')
    training_data = simulator_bf.sample_parallel(n_train_data)
    with open(data_path / f'train_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
else:
    try:
        with open(data_path / 'validation_data.pkl', 'rb') as f:
            validation_data = pickle.load(f)
        with open(data_path / 'train_data.pkl', 'rb') as f:
            training_data = pickle.load(f)

    except FileNotFoundError:
        logging.warning('No data loaded.')

#%%
param_names = ['prevalence_true', 'prevalence_subsample']
param_names_pretty = ['True prevalence', 'Subsample prevalence']
adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    # observables
    .as_set('data')
    .concatenate('data', into="summary_variables")
    # parameters
    .constrain(param_names, lower=0, upper=1)
    .concatenate(param_names, into='inference_variables')
    .keep(["inference_variables", "summary_variables"])
)

if job_array_id == -1:  # test run
    BATCH_SIZE = 64
    EPOCHS = 1
    summary_network = bf.networks.DeepSet(summary_dim=len(param_names))
    inference_network = bf.networks.CouplingFlow(depth=2)
    network_name = 'test'
    training_data = {}
    for k, v in validation_data.items():
        training_data[k] = v[:100]
elif job_array_id % 4 == 0:
    BATCH_SIZE = 64
    EPOCHS = 300
    summary_network = bf.networks.DeepSet(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1, widths=[128, 128]))
    network_name = 'FlowMatching_DeepSet'
else:
    raise ValueError("Invalid job_array_id.")


model_path = BASE / 'models' / f'prevalence_model_{network_name}.keras'
logging.info(model_path)

workflow = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=summary_network,
    inference_network=inference_network,
)
# %%
class HistoryClass(object):
    def __init__(self, history_to_save):
        self.history = history_to_save

if not os.path.exists(model_path):
    history = workflow.fit_offline(
        data=training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
    )
    workflow.approximator.save(filepath=model_path)

    diagnostics = workflow.compute_default_diagnostics(
        test_data=validation_data, num_samples=500, approximator_kwargs=dict(batch_size=BATCH_SIZE // 2),
        variable_names=param_names_pretty
    )
    logging.info(f"RMSE {diagnostics.loc['NRMSE'].mean()}")
    logging.info(f"Calibration Error {diagnostics.loc['Calibration Error'].mean()}")

    with open(BASE / 'models' / f'history_{network_name}.pkl', 'wb') as file:
        model_history = HistoryClass(history.history)
        pickle.dump(model_history, file)
else:
    workflow.approximator = keras.saving.load_model(filepath=model_path)

    try:
        with open(BASE / 'models' / f'history_{network_name}.pkl', 'rb') as file:
            workflow.history = pickle.load(file)
    except FileNotFoundError:
        logging.info("No history file found.")

#%%
diagnostics = workflow.plot_default_diagnostics(
    test_data=validation_data, num_samples=500, approximator_kwargs=dict(batch_size=BATCH_SIZE // 2),
    loss_kwargs=dict(val_color=colors[-1], train_color='black'),
    recovery_kwargs=dict(figsize=(5,2), add_corr=False, color=colors[-1], label_fontsize=9, metric_fontsize=9, tick_fontsize=9),
    calibration_ecdf_kwargs=dict(rank_ecdf_color=colors[-1]),
    coverage_kwargs=dict(color=colors[-1]),
    z_score_contraction_kwargs=dict(color=colors[-1]),
    variable_names=param_names_pretty
)
for diagnostic, fig_d in diagnostics.items():
    if diagnostic == 'recovery':
        fig_d.axes[0].set_title(None)
        fig_d.axes[1].set_title(None)
    fig_d.savefig(BASE / 'plots' / f'{network_name}_{diagnostic}.pdf', bbox_inches='tight')
    plt.close(fig_d)

#%%
def train_c2st(estimates, targets, n_random=10):
    """C2ST classifier on (posterior draw, data embedding) pairs plus label-permuted classifiers
    for the permutation p-value."""
    mean, std = np.mean(estimates, axis=0), np.std(estimates, axis=0)
    estimates = (estimates - mean) / std
    targets = (targets - mean) / std
    results = bf.diagnostics.metrics.classifier_two_sample_test(
        estimates=estimates, targets=targets, return_metric_only=False, batch_size=BATCH_SIZE, standardize=False
    )

    random_results = []
    full_set = np.concatenate((estimates, targets), axis=0)
    for _ in range(n_random):
        np.random.shuffle(full_set)  # permute all labels to create random classifier
        _random_results = bf.diagnostics.metrics.classifier_two_sample_test(
            estimates=full_set[:estimates.shape[0]], targets=full_set[estimates.shape[0]:],
            return_metric_only=False, batch_size=BATCH_SIZE, cross_validation_splits=0,
            validation_split=0.1, standardize=False
        )
        random_results.append(_random_results)
    return results, random_results, mean, std


def score_c2st(estimates_real, c2st):
    """Apply a trained C2ST to real-data (posterior draw, embedding) pairs.
    Returns per-draw scores, the test statistic and the permutation p-value."""
    results, random_results, mean, std = c2st
    estimates_real = (estimates_real - mean) / std
    scores = np.array([c.predict(estimates_real).flatten() for c in results['classifiers']])
    scores = np.maximum(scores, 1 - scores)
    score = np.mean(scores, axis=0)
    statistic = np.mean((score - 0.5) ** 2)
    scores_random = np.array([c['classifiers'][0].predict(estimates_real).flatten() for c in random_results])
    scores_random = np.maximum(scores_random, 1 - scores_random)
    statistic_random = np.mean((scores_random - 0.5) ** 2, axis=-1)
    return score, statistic, np.mean(statistic_random > statistic)


logging.info('Prepare C2ST classifier (NPE)')
embedded_data = workflow.approximator.summarize(validation_data)
targets = np.concatenate((validation_data['prevalence_true'], embedded_data), axis=-1)
posterior_samples_test = workflow.sample(conditions=validation_data, num_samples=1,
                                         batch_size=BATCH_SIZE // 2)
estimates = np.concatenate((posterior_samples_test['prevalence_true'][:, 0], embedded_data), axis=-1)
c2st_npe = train_c2st(estimates, targets)


mrp_file = BASE / 'models' / 'mrp_results.pkl'
if mrp_file.exists():
    with open(mrp_file, 'rb') as f:
        mrp_cache = pickle.load(f)
else:
    mrp_cache = {}

def _fit_mrp_validation(arr, seed):
    """fit MRP on the stored validation datasets."""
    # Inverse of convert_to_nn_input
    df = pd.DataFrame(arr, columns=['y_test'] + list(cat_to_int.keys()))
    df = df.mask(df == -1)  # -1 encodes missing
    for cat, mapping in cat_to_int.items():
        df[cat] = df[cat].map({v: k for k, v in mapping.items()})
    df['birth_country'] = df['birth_country'].astype(float)
    # fitting
    return fit_mrp(df, chains=1, seed=seed)['prevalence']


logging.info('Prepare C2ST classifier (MRP)')
if 'valid' not in mrp_cache:
    mrp_cache['valid'] = Parallel(n_jobs=n_cpus, verbose=1)(
        delayed(_fit_mrp_validation)(d_i, i) for i, d_i in enumerate(validation_data['data'])
    )
    with open(mrp_file, 'wb') as f:
        pickle.dump(mrp_cache, f)
mrp_valid_draw = np.array([p[np.random.randint(len(p))] for p in mrp_cache['valid']])[:, None]
estimates_mrp = np.concatenate((mrp_valid_draw, embedded_data), axis=-1)
c2st_mrp = train_c2st(estimates_mrp, targets)  # own classifier, same NPE embedding of the data

#%%
def error(a, b):
    """Error in percentage points"""
    return np.abs(a - b)*100

logging.info('Inference on test data...')
num_samples = 500

results_missing = {
    'unadjusted': np.full((5, n_val_data), np.nan),
    'MRP': np.full((5, n_val_data), np.nan),
    'NPE': np.full((5, n_val_data), np.nan)
}
results = {
    'unadjusted': np.full((5, n_val_data), np.nan),
    'MRP': np.full((5, n_val_data), np.nan),
    'NPE': np.full((5, n_val_data), np.nan)
}

for e_index in range(1, 6):
    logging.info(f"\n--- Epoch T{e_index} ---")
    sim_out_missing = Parallel(n_jobs=n_cpus)(
        delayed(simulate_population)(
            epoch_index=e_index,
            n_out=int(full_population_size*0.1),
            with_missingness=True,
            bootstrap_resamples=0,
            seed=i_test+e_index
        ) for i_test in range(n_val_data)
    )
    # simulate same datasets
    sim_out = Parallel(n_jobs=n_cpus)(
        delayed(simulate_population)(
            epoch_index=e_index,
            n_out=int(full_population_size * 0.1),
            with_missingness=False,
            bootstrap_resamples=0,
            seed=i_test+e_index
        ) for i_test in range(n_val_data)
    )

    data_list = []
    for i_test in range(n_val_data):
        data_df = sim_out_missing[i_test]['subsample']  # NPE is only trained on data with missing entries
        data = convert_to_nn_input(data_df)
        data_list.append(data)
    data_list = np.stack(data_list, axis=0)
    posterior_samples_real = workflow.sample(conditions={'data': data_list}, num_samples=num_samples,
                                             batch_size=BATCH_SIZE)

    # Posterior median
    posterior_median = np.median(posterior_samples_real['prevalence_true'], axis=1).flatten()

    results_missing['unadjusted'][e_index-1] = error(
        np.array([pv['prevalence_subsample'] for pv in sim_out_missing]),
        np.array([pv['prevalence_true'] for pv in sim_out_missing])
    )
    results_missing['NPE'][e_index - 1] = error(
        posterior_median,
        np.array([pv['prevalence_true'] for pv in sim_out_missing])
    )
    results['unadjusted'][e_index - 1] = error(
        np.array([pv['prevalence_subsample'] for pv in sim_out]),
        np.array([pv['prevalence_true'] for pv in sim_out])
    )
    results['NPE'][e_index - 1] = error(
        posterior_median,
        np.array([pv['prevalence_true'] for pv in sim_out])
    )
    logging.info(f"Error NPE: {np.median(results['NPE'][e_index - 1])}")

    # MRP (Stan): outcome model + poststratification on the sample it rakes itself
    if 'sim' in mrp_cache:
        results_missing['MRP'][e_index - 1] = mrp_cache['sim_missing'][e_index - 1]
        results['MRP'][e_index - 1] = mrp_cache['sim'][e_index - 1]
    else:
        t0 = time.perf_counter()
        for res, sims in ((results_missing, sim_out_missing), (results, sim_out)):
            mrp_draws = Parallel(n_jobs=n_cpus, verbose=1)(
                delayed(fit_mrp)(pv['subsample'], seed=i, chains=1, iter_sampling=num_samples)
                for i, pv in enumerate(sims[:n_val_data])
            )
            res['MRP'][e_index - 1] = error(
                np.array([np.median(d['prevalence']) for d in mrp_draws]),
                np.array([pv['prevalence_true'] for pv in sims[:n_val_data]])
            )
        logging.info(f"Time for MRP: {time.perf_counter() - t0:.2f}s")
    logging.info(f"Error MRP: {np.median(results['MRP'][e_index - 1])}")

if 'sim' not in mrp_cache:
    mrp_cache['sim_missing'] = results_missing['MRP']
    mrp_cache['sim'] = results['MRP']
    with open(mrp_file, 'wb') as f:
        pickle.dump(mrp_cache, f)

#%%
for missing, result in zip([True, False], [results_missing, results]):
    # Plot boxplot
    fig, ax = plt.subplots(figsize=(5, 2), layout='constrained')
    width = 0.2
    offsets = [-width, 0, width]
    labels = ['Unadjusted', 'MRP', 'Bias-aware NPE']
    for i, samples in enumerate(result.values()):
        pos = np.arange(5) + offsets[i]
        bp = ax.boxplot(
            [samples[t].flatten() for t in range(5)],
            #[samples[t][~np.isnan(samples[t])] for t in range(5)],
            positions=pos,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black"),
            boxprops=dict(facecolor=colors[i], alpha=0.7),
            whiskerprops=dict(color=colors[i]),
            capprops=dict(color=colors[i]),
        )
        bp["boxes"][0].set_label(labels[i])

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$'], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    if missing:
        ax.set_xlabel(r'Simulated round (with missingness)', fontsize=15)
    else:
        ax.set_xlabel(r'Simulated round', fontsize=15)
    ax.set_ylabel(r'Absolute error ($\%$)', fontsize=15)
    ax.grid(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 3)
    if not missing:
        fig.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=15)
    if missing:
        fig.savefig(BASE / 'plots' / f'{network_name}_koco19_prevalence_missing.pdf', bbox_inches='tight')
    else:
        fig.savefig(BASE / 'plots' / f'{network_name}_koco19_prevalence.pdf', bbox_inches='tight')
    plt.show()

#%%
logging.info('Inference on real data...')
real_data_path = lambda epoch_index: f'data/koco19_T{epoch_index}_prepared.csv'
results_real = {
    'unadjusted': [],
    'MRP': [],
    'NPE': [],
    'C2ST': [],
    'C2ST_MRP': [],
}
missing_round = []
c2st_result_real_random = []
c2st_result_real_random_mrp = []
mrp_fits_real = []
num_samples = 1000

for e_index in range(1, 6):
    logging.info(f"\n--- Epoch T{e_index} ---")
    sim_out = simulate_population(  # no simulation, real outcomes used
        epoch_index=e_index,
        n_out=int(full_population_size),
        real_data_path=real_data_path(e_index),
        bootstrap_resamples=100,
    )
    results_real['unadjusted'].append(sim_out['prevalence_subsample'])

    sim_out = simulate_population(  # no simulation, real outcomes used
        epoch_index=e_index,
        n_out=int(full_population_size),
        real_data_path=real_data_path(e_index),
        bootstrap_resamples=0,  # so the real df is returned
    )
    real_data_df = sim_out['subsample']
    real_data = convert_to_nn_input(real_data_df)
    posterior_samples_real = workflow.sample(conditions={'data': real_data[None]}, num_samples=num_samples,
                                             batch_size=BATCH_SIZE // 2)
    results_real['NPE'].append(posterior_samples_real['prevalence_true'])
    missing_round.append(sum(pd.isna(real_data_df['y_test'])) / len(real_data_df) * 100)

    if 'real' in mrp_cache:
        mrp_fits_real.append(mrp_cache['real'][e_index - 1])
    else:
        mrp_fits_real.append(fit_mrp(real_data_df, seed=e_index, iter_sampling=num_samples))
    results_real['MRP'].append(mrp_fits_real[-1]['prevalence'])
    logging.info(f"MRP median: {np.median(results_real['MRP'][-1]) * 100:.2f}%")

    # use C2ST to evaluate posterior samples quality: NPE and MRP, each with its own classifier
    embedded_real_data = workflow.approximator.summarize({'data': real_data[None]})
    embedded_real_data = np.repeat(embedded_real_data, repeats=num_samples, axis=0)
    estimates_real = np.concatenate((posterior_samples_real['prevalence_true'][0], embedded_real_data), axis=-1)
    c2st_score, test_statistic, p_val = score_c2st(estimates_real, c2st_npe)
    results_real['C2ST'].append(c2st_score)
    c2st_result_real_random.append((test_statistic, p_val))
    logging.info(f'NPE C2ST Accuracy: {np.mean(c2st_score)}, Statistic: {test_statistic}, p-value: {p_val}')

    mrp_draws = np.random.default_rng(e_index).choice(mrp_fits_real[-1]['prevalence'], size=num_samples,
                                                      replace=False)
    estimates_real_mrp = np.concatenate((mrp_draws[:, None], embedded_real_data), axis=-1)
    c2st_score_mrp, test_statistic_mrp, p_val_mrp = score_c2st(estimates_real_mrp, c2st_mrp)
    results_real['C2ST_MRP'].append(c2st_score_mrp)
    c2st_result_real_random_mrp.append((test_statistic_mrp, p_val_mrp))
    logging.info(f'MRP C2ST Accuracy: {np.mean(c2st_score_mrp)}, Statistic: {test_statistic_mrp}, '
                 f'p-value: {p_val_mrp}')

if 'real' not in mrp_cache:
    mrp_cache['real'] = mrp_fits_real
    with open(mrp_file, 'wb') as f:
        pickle.dump(mrp_cache, f)

#%%
# Plot violin plot
fig, ax = plt.subplots(figsize=(5, 2), layout='constrained')
width = 0.2
offsets = [-width, 0, width]
labels = ['Unadjusted', 'Poststratification', 'Bias-aware NPE']

for i, samples in enumerate([results_real[k] for k in ('unadjusted', 'MRP', 'NPE')]):
    pos = np.arange(5) + offsets[i]
    parts = ax.violinplot(
        [samples[t].flatten()*100 for t in range(5)],
        positions=pos,
        widths=0.15,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    # Label each set of violin bodies for legend
    for body in parts['bodies']:
        body.set_color(colors[i])
        body.set_alpha(0.65)
        body.set_edgecolor('black')
    parts['cmedians'].set_color('black')
    for body in parts['bodies']:
        body.set_label(labels[i])
        break
    logging.info(f"{labels[i]}: Median {np.median([samples[t].flatten()*100 for t in range(5)], axis=1)}%")

ax.set_xticks(np.arange(5))
ax.set_xticklabels([r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$'], fontsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.set_xlabel(r'KoCo19 Round', fontsize=15)
ax.set_ylabel('Estimated\nPrevalence'+r' ($\%$)', fontsize=15)
ax.set_ylim(0,15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis='y')
ax.set_ylim(0, None)
#fig.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=15)
fig.savefig(BASE / 'plots' / f'{network_name}_koco19_prevalence_real.pdf', bbox_inches='tight')
plt.show()

#%%
class ScalarFormatter1f(plt.ScalarFormatter):
    def _set_format(self):
        self.format = '%1.1f'

def plot_c2st_histograms(posteriors, c2st_scores, c2st_tests, save_path, bins=20):
    """Per-round posterior histograms of the real-data prevalence, bins coloured by mean C2ST score."""
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Reds_trunc",
        plt.cm.Reds(np.linspace(0.1, 1.0, 256))
    )
    fig, ax = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(10, 2), layout='constrained')
    for epoch_idx in range(1, 6):
        # compute bin assignment
        x = np.asarray(posteriors[epoch_idx-1]).flatten() * 100
        counts, bin_edges = np.histogram(x, bins=bins, density=True)
        bin_idx = np.digitize(x, bin_edges) - 1

        # compute mean color per bin
        bin_color = np.array([
            np.mean(c2st_scores[epoch_idx-1][bin_idx == i]) if np.any(bin_idx == i) else 0
            for i in range(bins)
        ])

        # plot histogram manually
        for i in range(bins):
            ax[epoch_idx-1].bar(
                bin_edges[i],
                counts[i],
                width=bin_edges[i + 1] - bin_edges[i],
                align="edge",
                color=cmap(norm(bin_color[i])),
                edgecolor=cmap(norm(bin_color[i]))
            )

        ax[epoch_idx-1].set_xlabel(r"Prevalence ($\%$)", fontsize=15)
        if epoch_idx == 1:
            ax[epoch_idx-1].set_ylabel("Posterior Density", fontsize=15)
        ax[epoch_idx-1].set_title(rf"Round $R_{epoch_idx}$", fontsize=15)
        # remove top and right spines
        ax[epoch_idx-1].spines['top'].set_visible(False)
        ax[epoch_idx-1].spines['right'].set_visible(False)
        # plot scores
        m_score = np.mean(c2st_scores[epoch_idx-1])
        ax[epoch_idx-1].text(
            0.95, 0.95,
            f"Mean C2ST={m_score:.2f}\np-value={c2st_tests[epoch_idx-1][1]:.2f}",
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax[epoch_idx-1].transAxes,
            fontsize=13,
        )
        formatter = ScalarFormatter1f(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax[epoch_idx-1].yaxis.set_major_formatter(formatter)
        ax[epoch_idx-1].tick_params(axis='y', labelsize=13)
        ax[epoch_idx-1].tick_params(axis='x', labelsize=13)

    # add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), fraction=0.02)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("C2ST score\n(mean per bin)", fontsize=15)
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()


plot_c2st_histograms(results_real['NPE'], results_real['C2ST'], c2st_result_real_random,
                     BASE / 'plots' / f'{network_name}_koco19_prevalence_real_histograms.pdf')
# MRP: the 1000 draws scored by its own classifier (subset drawn with the same seed as in the loop)
mrp_scored_draws = [np.random.default_rng(e).choice(mrp_fits_real[e-1]['prevalence'], size=num_samples, replace=False)
                    for e in range(1, 6)]
plot_c2st_histograms(mrp_scored_draws, results_real['C2ST_MRP'], c2st_result_real_random_mrp,
                     BASE / 'plots' / f'{network_name}_koco19_prevalence_real_histograms_mrp.pdf')

logging.info('Done.')

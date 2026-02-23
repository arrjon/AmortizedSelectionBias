#%%
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
from tqdm import tqdm

import keras
import bayesflow as bf

from KoCo.prevalence_simulate import simulate_population, full_population_size

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
#for k, v in simulator_bf.sample_parallel(100).items():
#    print(k, v.shape if isinstance(v, np.ndarray) else type(v))

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
logging.info('Prepare C2ST classifier')
embedded_data = workflow.approximator.summarize(validation_data)
targets = np.concatenate((validation_data['prevalence_true'], embedded_data), axis=-1)
posterior_samples_test = workflow.sample(conditions=validation_data, num_samples=1,
                                         batch_size=BATCH_SIZE // 2)
estimates = np.concatenate((posterior_samples_test['prevalence_true'][:, 0], embedded_data), axis=-1)
estimates_mean = np.mean(estimates, axis=0)
estimates_std = np.std(estimates, axis=0)
estimates = (estimates - estimates_mean) / estimates_std
targets = (targets - estimates_mean) / estimates_std
logging.info('Train C2ST classifier')
c2st_results = bf.diagnostics.metrics.classifier_two_sample_test(
    estimates=estimates,
    targets=targets,
    return_metric_only=False,
    batch_size=BATCH_SIZE,
    standardize=False
)
logging.info(f'C2ST Accuracy: {c2st_results["score"]}')

logging.info('Train C2ST random classifiers')
c2st_results_random = []
full_set = np.concatenate((estimates, targets), axis=0)
for _ in range(10):
    # permute all labels to create random classifier
    np.random.shuffle(full_set)
    estimates_random = full_set[:estimates.shape[0]]
    targets_random = full_set[estimates.shape[0]:]

    c2st_results_random.append(bf.diagnostics.metrics.classifier_two_sample_test(
        estimates=estimates_random,
        targets=targets_random,
        return_metric_only=False,
        batch_size=BATCH_SIZE,
        cross_validation_splits=0,
        validation_split=0.1,
        standardize=False
    ))

#%%
def error(a, b):
    """Error in percentage points"""
    return np.abs(a - b)*100

logging.info('Inference on test data...')
num_samples = 500

results_missing = {
    'unadjusted': np.zeros((5, n_val_data)),
    'adjusted': np.zeros((5, n_val_data)),
    'NPE': np.zeros((5, n_val_data))
}
results = {
    'unadjusted': np.zeros((5, n_val_data)),
    'adjusted': np.zeros((5, n_val_data)),
    'NPE': np.zeros((5, n_val_data))
}
for e_index in range(1, 6):
    logging.info(f"\n--- Epoch T{e_index} ---")
    sim_out_missing = Parallel(n_jobs=n_cpus)(
        delayed(simulate_population)(
            epoch_index=e_index,
            n_out=int(full_population_size*0.1),
            use_real_outcomes=False,
            with_missingness=True,
            bootstrap_resamples=0,
            seed=i_test
        ) for i_test in range(n_val_data)
    )
    # simulate same datasets
    sim_out = Parallel(n_jobs=n_cpus)(
        delayed(simulate_population)(
            epoch_index=e_index,
            n_out=int(full_population_size * 0.1),
            use_real_outcomes=False,
            with_missingness=False,
            bootstrap_resamples=0,
            seed=i_test
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

    logging.info('Compute posterior modes...')
    posterior_mode = []
    for i in tqdm(range(n_val_data)):  # batch the data for log prob computation
        batch = {
            'data': np.repeat(data_list[i][None], num_samples, axis=0),
            'prevalence_true': posterior_samples_real['prevalence_true'][i],
            'prevalence_subsample': posterior_samples_real['prevalence_subsample'][i]
        }
        log_prob = workflow.log_prob(batch)
        mode_idx = np.argmax(log_prob)
        posterior_mode.append(batch['prevalence_true'][mode_idx].item())

    results_missing['unadjusted'][e_index-1] = error(
        np.array([pv['prevalence_subsample'] for pv in sim_out_missing]),
        np.array([pv['prevalence_true'] for pv in sim_out_missing])
    )
    results_missing['adjusted'][e_index - 1] = error(
        np.array([pv['prevalence_subsample_weighted'] for pv in sim_out_missing]),
        np.array([pv['prevalence_true'] for pv in sim_out_missing])
    )
    results_missing['NPE'][e_index - 1] = error(
        #np.median(posterior_samples_real['prevalence_true'], axis=1).flatten(),
        np.array(posterior_mode),
        np.array([pv['prevalence_true'] for pv in sim_out_missing])
    )
    results['unadjusted'][e_index - 1] = error(
        np.array([pv['prevalence_subsample'] for pv in sim_out]),
        np.array([pv['prevalence_true'] for pv in sim_out])
    )
    results['adjusted'][e_index - 1] = error(
        np.array([pv['prevalence_subsample_weighted'] for pv in sim_out]),
        np.array([pv['prevalence_true'] for pv in sim_out])
    )
    results['NPE'][e_index - 1] = error(
        #np.median(posterior_samples_real['prevalence_true'], axis=1).flatten(),
        np.array(posterior_mode),
        np.array([pv['prevalence_true'] for pv in sim_out])
    )
    logging.info(f"Error NPE: {np.median(results['NPE'][e_index - 1])}")

for missing, result in zip([True, False], [results_missing, results]):
    # Plot boxplot
    fig, ax = plt.subplots(figsize=(5, 2), layout='constrained')
    width = 0.2
    offsets = [-width, 0, width]
    labels = ['Unadjusted', 'Weighted', 'Bias-aware NPE']
    for i, samples in enumerate(result.values()):
        pos = np.arange(5) + offsets[i]
        bp = ax.boxplot(
            [samples[t].flatten() for t in range(5)],
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
    ax.set_xticklabels(['$R_1$', '$R_2$', '$R_3$', '$R_4$', '$R_5$'])
    if missing:
        ax.set_xlabel(r'Simulated round (with missingness)')
    else:
        ax.set_xlabel(r'Simulated round')
    ax.set_ylabel('Absolute error\nin percentage points')
    ax.grid(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 3)
    fig.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.15), frameon=False)
    if missing:
        fig.savefig(BASE / 'plots' / f'{network_name}_koco19_prevalence_missing.pdf', bbox_inches='tight')
    else:
        fig.savefig(BASE / 'plots' / f'{network_name}_koco19_prevalence.pdf', bbox_inches='tight')
    plt.show()

#%%
logging.info('Inference on real data...')
results_real = {
    'unadjusted': [],
    'adjusted': [],
    'NPE': [],
    'C2ST': [],
}
missing_round = []
c2st_result_real_random = []
num_samples = 1000

for e_index in range(1, 6):
    logging.info(f"\n--- Epoch T{e_index} ---")
    sim_out = simulate_population(  # no simulation, real outcomes used
        epoch_index=e_index,
        n_out=int(full_population_size),
        use_real_outcomes=True,
        bootstrap_resamples=100,
    )
    results_real['unadjusted'].append(sim_out['prevalence_subsample'])
    results_real['adjusted'].append(sim_out['prevalence_subsample_weighted'])

    sim_out = simulate_population(  # no simulation, real outcomes used
        epoch_index=e_index,
        n_out=int(full_population_size),
        use_real_outcomes=True,
        bootstrap_resamples=0,  # so the real df is returned
    )
    real_data_df = sim_out['subsample']
    real_data = convert_to_nn_input(real_data_df)
    posterior_samples_real = workflow.sample(conditions={'data': real_data[None]}, num_samples=num_samples,
                                             batch_size=BATCH_SIZE // 2)
    results_real['NPE'].append(posterior_samples_real['prevalence_true'])
    missing_round.append(sum(pd.isna(real_data_df['y_test'])) / len(real_data_df) * 100)

    # use C2ST to evaluate posterior samples quality
    embedded_real_data = workflow.approximator.summarize({'data': real_data[None]})
    embedded_real_data = np.repeat(embedded_real_data, repeats=num_samples, axis=0)
    estimates_real = np.concatenate((posterior_samples_real['prevalence_true'][0], embedded_real_data), axis=-1)
    estimates_real = (estimates_real - estimates_mean) / estimates_std
    scores = np.array([c.predict(estimates_real).flatten() for c in c2st_results['classifiers']])
    scores = np.maximum(scores, 1 - scores)
    c2st_score = np.mean(scores, axis=0)
    test_statistic = np.mean((c2st_score - 0.5) ** 2)
    results_real['C2ST'].append(c2st_score)
    logging.info(f'C2ST Accuracy: {np.mean(results_real["C2ST"][-1])}')

    # apply random classifiers
    scores_random = np.array([c['classifiers'][0].predict(estimates_real).flatten() for c in c2st_results_random])
    scores_random = np.maximum(scores_random, 1 - scores_random)
    test_statistic_random = np.mean((scores_random - 0.5) ** 2, axis=-1)
    p_val = np.mean(test_statistic_random > test_statistic)
    c2st_result_real_random.append((test_statistic, p_val))
    logging.info(f'C2ST Statistic: {test_statistic}, p-value: {p_val}')

#%%
# Plot violin plot
fig, ax = plt.subplots(figsize=(5, 2), layout='constrained')
width = 0.2
offsets = [-width, 0, width]
labels = ['Unadjusted', 'Weighted', 'Bias-aware NPE']

for i, samples in enumerate(list(results_real.values())[:-1]):  # exclude C2ST
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
    parts['cmedians'].set_color(colors[i])
    for body in parts['bodies']:
        body.set_label(labels[i])
        break
    logging.info(f"{labels[i]}: Median {np.median([samples[t].flatten()*100 for t in range(5)], axis=1)}%")

ax.set_xticks(np.arange(5))
ax.set_xticklabels([f'$R_1$', f'$R_2$',
                    f'$R_3$', f'$R_4$',
                    f'$R_5$'])
ax.set_xlabel(r'KoCo19 Round')
ax.set_ylabel('Estimated\nPrevalence'+r' ($\%$)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y')
ax.set_ylim(0, None)
fig.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.15), frameon=False)
fig.savefig(BASE / 'plots' / f'{network_name}_koco19_prevalence_real.pdf', bbox_inches='tight')
plt.show()

#%%
bins = 20
norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "Reds_trunc",
    plt.cm.Reds(np.linspace(0.1, 1.0, 256))
)
fig, ax = plt.subplots(nrows=1, ncols=5, sharey=True, sharex=True, figsize=(10, 2), layout='constrained')
for epoch_idx in range(1, 6):
    # compute bin assignment
    x = results_real['NPE'][epoch_idx-1].flatten() * 100
    counts, bin_edges = np.histogram(x, bins=bins, density=True)
    bin_idx = np.digitize(x, bin_edges) - 1

    # compute mean color per bin
    bin_color = np.array([
        np.mean(results_real['C2ST'][epoch_idx-1][bin_idx == i]) if np.any(bin_idx == i) else 0
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

    ax[epoch_idx-1].set_xlabel(r"Prevalence ($\%$)")
    if epoch_idx == 1:
        ax[epoch_idx-1].set_ylabel("Posterior Density")
    ax[epoch_idx-1].set_title(rf"Round $R_{epoch_idx}$")
    # remove top and right spines
    ax[epoch_idx-1].spines['top'].set_visible(False)
    ax[epoch_idx-1].spines['right'].set_visible(False)
    # plot scores
    m_score = np.mean(results_real['C2ST'][epoch_idx-1])
    ax[epoch_idx-1].text(
        0.95, 0.95,
        f"Mean C2ST={m_score:.2f}\np-value={c2st_result_real_random[epoch_idx-1][1]:.1f}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[epoch_idx-1].transAxes,
        fontsize=9,
    )

# add colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), label="C2ST score\n(mean per bin)", fraction=0.02)
fig.savefig(BASE / 'plots' / f'{network_name}_koco19_prevalence_real_histograms.pdf', bbox_inches='tight')
plt.show()

logging.info('Done.')

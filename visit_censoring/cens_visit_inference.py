# %%  uv pip install "git+https://github.com/bayesflow-org/bayesflow.git@5156f63"
import os
os.environ['KERAS_BACKEND'] = 'jax'

import logging
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as mad

import keras
import bayesflow as bf

from visit_censoring.cens_visit_plotting import plot_params, plot_cumhaz, colors, plot_params_error, plot_hazard_nrmse, plot_data_summaries
from visit_censoring.cens_visit_helper import extract_batches_to_dict, compute_gamma_params

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path('/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/visit_censoring')
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 2))  # 2, 6
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))
partition = os.environ.get('SLURM_JOB_PARTITION', 'local')

if 'gpu' in partition or 'intelsr' in partition:
    simulate_all_epochs = lambda: 0
else:
    from visit_censoring.cens_visit_simulate import simulate_all_epochs
posterior_samples_model = {}

# %%
epochs = ["epoch1", "epoch2", "epoch3", "epoch4"]
data_names = ['illt', 'ills', 'dt', 'ds', 'sex', 'age']
params_beta = ['beta01_age', 'beta02_age', 'beta12_age', 'beta01_sex', 'beta02_sex', 'beta12_sex']
param_names_pretty = [r'$a_{01}$', r'$a_{02}$', r'$a_{12}$',
                      r'$\kappa_{01}$', r'$\kappa_{02}$', r'$\kappa_{12}$',
                      r'$\beta_{01}^\text{sex}$', r'$\beta_{02}^\text{sex}$', r'$\beta_{12}^\text{sex}$',
                      r'$\beta_{01}^\text{age}$', r'$\beta_{02}^\text{age}$', r'$\beta_{12}^\text{age}$']

# load baseline results
with open(BASE / 'baseline.json') as f:
    _baseline = json.load(f)

baseline = {e: {} for e in epochs}
for e in epochs:
    baseline[e] = _baseline[e]
    b = _baseline[e]['coefs']
    for p in params_beta:
        for k in b:
            if (k['row'] == f"{p[4:].split('_')[1]}_{p[4:].split('_')[0]}" or
                k['row'] == f"{p[4:].split('_')[1]}Centered_{p[4:].split('_')[0]}"):
                baseline[e][p] = {
                    'naive_cox': k['naive_cox'],
                    'weibull': k['idm_weib'],
                    'splines': k['idm_splines']
                }

# Framingham epoch priors
cv = 1.0
std_min = 1.0
shape_mean = 1.0  # exponential as baseline
shape_cv = 0.25
shape_params = compute_gamma_params(shape_mean, cv=shape_cv)
a_mean = 0.0003
priors = {
    'a01': compute_gamma_params(a_mean, cv=cv),
    'a02': compute_gamma_params(a_mean, cv=cv),
    'a12': compute_gamma_params(a_mean, cv=cv),
    'shape01': shape_params,
    'shape02': shape_params,
    'shape12': shape_params,
    'beta01_sex': {'mean': 0.0, 'sd': std_min},
    'beta02_sex': {'mean': 0.0, 'sd': std_min},
    'beta12_sex': {'mean': 0.0, 'sd': std_min},
    'beta01_age': {'mean': 0.0, 'sd': std_min},
    'beta02_age': {'mean': 0.0, 'sd': std_min},
    'beta12_age': {'mean': 0.0, 'sd': std_min},
}
param_names = list(priors.keys())

# %%
data_path = BASE / 'data'
n_val_data = 1000
n_train_data = 20000
create_data = False
training_data, validation_data = None, None
training_data_full, validation_data_full = None, None

if create_data:
    # Simulate and extract structured arrays
    logging.info('Generate validation data...')
    df_valid = simulate_all_epochs(priors=priors, n_replicates=n_val_data // 4, ncores=n_cpus)
    validation_data = extract_batches_to_dict(df_valid, scheme='CensVisit')
    with open(data_path / 'validation_data.pkl', 'wb') as f:
       pickle.dump(validation_data, f)
    validation_data_full = extract_batches_to_dict(df_valid, scheme='Full')
    with open(data_path / 'validation_data_full.pkl', 'wb') as f:
        pickle.dump(validation_data_full, f)

    logging.info('Generate training data...')
    for i in range(10):
        logging.info(f'  Batch {i+1}/10')
        df = simulate_all_epochs(priors=priors, n_replicates=n_train_data // 4 // 10, ncores=n_cpus)
        training_data = extract_batches_to_dict(df, scheme='CensVisit')
        with open(data_path / f'train_data_{i}.pkl', 'wb') as f:
            pickle.dump(training_data, f)
        training_data_full = extract_batches_to_dict(df, scheme='Full')
        with open(data_path / f'train_data_{i}_full.pkl', 'wb') as f:
            pickle.dump(training_data_full, f)
else:
    try:
        with open(data_path / 'validation_data.pkl', 'rb') as f:
            validation_data = pickle.load(f)
        with open(data_path / 'validation_data_full.pkl', 'rb') as f:
            validation_data_full = pickle.load(f)

        if os.path.exists(data_path / 'train_data.pkl'):
            with open(data_path / 'train_data.pkl', 'rb') as f:
                training_data = pickle.load(f)
            with open(data_path / 'train_data_full.pkl', 'rb') as f:
                training_data_full = pickle.load(f)
        else:
            logging.info('load single files and combine into one training data file...')
            for add_name in ['', '_full']:
                training_data = {k: [] for k in validation_data}
                for i in range(10):
                    with open(data_path / f'train_data_{i}{add_name}.pkl', 'rb') as f:
                        file = pickle.load(f)
                    for k in file:
                        training_data[k].append(file[k])
                for k in training_data:
                    training_data[k] = np.vstack(training_data[k])
                with open(data_path / f'train_data{add_name}.pkl', 'wb') as f:
                    pickle.dump(training_data, f)

    except FileNotFoundError:
        logging.warning('No data loaded.')

for data in [validation_data_full, training_data_full]:
    if data is not None:  # censoring times capped at 5 years
        data['ds'][data['dt'] > 1825] = 0
        data['dt'][data['dt'] > 1825] = 1825
        data['ills'][data['illt'] > 1825] = 0
        data['illt'][data['illt'] > 1825] = 1825

# %%
adapter = (
    bf.adapters.Adapter()
    .to_array()
    .standardize('epoch', mean=1, std=1)
    .convert_dtype(include='epoch', from_dtype="float64", to_dtype="int")
    .one_hot('epoch', num_classes=4)
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    # observables
    .constrain(['illt', 'dt'], lower=-2, upper=1826, inclusive='both')
    .standardize(['illt', 'dt'], mean=-25, std=20)
    .standardize('age', mean=0, std=7.5)
    .expand_dims(data_names, axis=-1)  # expand patients
    .concatenate(data_names + ['epoch'], into="summary_variables", axis=2)
    # parameters
    .constrain(['a01', 'a02', 'a12'], lower=0, upper=0.01)
    .standardize(['a01', 'a02', 'a12'], mean=-6, std=2)
    .log(['shape01', 'shape02', 'shape12'])
    .concatenate(param_names, into="inference_variables")
    .keep(["inference_variables", "summary_variables"])
)

if job_array_id == -1:  # test run
    BATCH_SIZE = 64
    EPOCHS = 2
    summary_network = bf.networks.DeepSet(summary_dim=len(param_names))
    inference_network = bf.networks.CouplingFlow(depth=2)
    network_name = 'test'
    training_data = {}
    for k, v in validation_data.items():
        training_data[k] = v[:100]
elif job_array_id % 4 == 0:
    BATCH_SIZE = 64
    EPOCHS = 300
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))
    network_name = 'FlowMatching_SetTransformer'
elif job_array_id % 4 == 1:
    BATCH_SIZE = 64
    EPOCHS = 300
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.DiffusionModel(subnet_kwargs=dict(dropout=0.1))
    network_name = 'DiffusionModel_SetTransformer'
elif job_array_id % 4 == 2:
    BATCH_SIZE = 64
    EPOCHS = 300
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.ConsistencyModel(total_steps=EPOCHS * (n_train_data // BATCH_SIZE),
                                                     subnet_kwargs=dict(dropout=0.1))
    network_name = 'ConsistencyModel_SetTransformer'
elif job_array_id % 4 == 3:
    BATCH_SIZE = 64
    EPOCHS = 100
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.CouplingFlow(depth=7, transform='spline',
                                                 subnet_kwargs=dict(dropout=0.1))
    network_name = 'CouplingFlow_SetTransformer'
else:
    raise ValueError("Invalid job_array_id.")

if job_array_id > 3:
    network_name += '_full'

model_path = BASE / 'models' / f'weibull_npe_model_{network_name}.keras'
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

    if 'full' in network_name:
        training_data = training_data_full
        validation_data = validation_data_full

    history = workflow.fit_offline(
        data=training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
    )
    workflow.approximator.save(filepath=model_path)

    diagnostics = workflow.compute_default_diagnostics(
        test_data=validation_data, num_samples=500, approximator_kwargs=dict(batch_size=BATCH_SIZE // 2)
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
        fig = bf.diagnostics.loss(workflow.history, val_color=colors[-1], train_color='black')
        fig.savefig(BASE / 'plots' / f'{network_name}_loss.pdf', bbox_inches='tight')
        plt.close()
    except FileNotFoundError:
        logging.info("No history file found.")

#%%
if network_name not in posterior_samples_model:
    logging.info('Sample parameters to compare models...')
    # custom plot for recovery for multiple models
    posterior_samples_valid = workflow.sample(conditions=validation_data, num_samples=1000,
                                              batch_size=BATCH_SIZE // 2)
    posterior_samples_model[f'{network_name}'] = posterior_samples_valid

    if 'full' in network_name:
        posterior_samples_valid = workflow.sample(conditions=validation_data_full, num_samples=1000,
                                                  batch_size=BATCH_SIZE // 2)
        posterior_samples_model[f'{network_name}-uncensored'] = posterior_samples_valid

if len(posterior_samples_model) == 3:
    logging.info('Compare models...')
    labels_dict = {
        'ConsistencyModel_SetTransformer': ('Bias-aware NPE (Observed Data)', colors[-1], 3),
        'ConsistencyModel_SetTransformer_full': ('NPE (Observed Data)', colors[2], 2),
        'ConsistencyModel_SetTransformer_full-uncensored': ('NPE (Full Data)', colors[0], 1)
    }
    labels = [labels_dict[m][0] for m in posterior_samples_model]
    colors_ordered = [labels_dict[m][1] for m in posterior_samples_model]
    posterior_samples_model = dict(sorted(posterior_samples_model.items(), key=lambda item: labels_dict[item[0]][2]))
    plot_params_error(
        posterior_samples_model,
        validation_data, validation_data_full,
        labels, param_names, param_names_pretty, colors_ordered,
        save_path=BASE / 'plots' / f'recovery_boxplot.pdf'
    )
    plot_hazard_nrmse(
        posterior_samples_model,
        validation_data, validation_data_full,
        labels_dict,
        save_path=BASE / 'plots' / f'hazard_nrmse.pdf'
    )

# %%
logging.info('Validation diagnostics...')
if 'full' in network_name:
    validation_data = validation_data_full

if not os.path.exists(BASE / 'plots' / f'{network_name}_recovery.pdf'):
    posterior_samples_valid = workflow.sample(conditions=validation_data, num_samples=1000,
                                              batch_size=BATCH_SIZE // 2)
    fig = bf.diagnostics.recovery(posterior_samples_valid, validation_data,
                                  color=colors[-1] if not 'full' in network_name else colors[2],
                                  variable_names=param_names_pretty)
    fig.savefig(BASE / 'plots' / f'{network_name}_recovery.pdf', bbox_inches='tight')
    plt.show()

    fig = bf.diagnostics.calibration_ecdf(posterior_samples_valid, validation_data,
                                          variable_names=param_names_pretty,
                                          rank_ecdf_color=colors[-1] if not 'full' in network_name else colors[2])
    ax = fig.get_axes()
    for a in ax:
        a.set_ylim(-0.25, 0.25)
    fig.savefig(BASE / 'plots' / f'{network_name}_ecdf.pdf', bbox_inches='tight')
    plt.show()

    _ = adapter.forward(validation_data)  # warm-up adapter
    ps_valid_adapted = adapter.forward(posterior_samples_valid, strict=False)
    valid_adapted = adapter.forward(validation_data, strict=False)
    fig = bf.diagnostics.recovery(ps_valid_adapted, valid_adapted,
                                  color=colors[-1] if not 'full' in network_name else colors[2],
                                  variable_names=[r'trans ' + p for p in param_names_pretty])
    fig.savefig(BASE / 'plots' / f'{network_name}_recovery_adapted.pdf', bbox_inches='tight')
    plt.show()

    vals = bf.diagnostics.posterior_contraction(posterior_samples_valid, validation_data)
    vals_mad = bf.diagnostics.posterior_contraction(posterior_samples_valid, validation_data, aggregation=mad)
    vals = {'contraction': vals['values'], 'mad': vals_mad['values']}
    df = pd.DataFrame(vals)
    df.index = param_names_pretty
    vals_tex = df.to_latex()
    with open(BASE / 'plots' / f'{network_name}_contraction.tex', 'a') as f:
        f.write(vals_tex)

logging.info('Prepare C2ST classifier')
embedded_data = []
for i in range(0, len(validation_data['dt']), BATCH_SIZE // 2):
    batch = {name: validation_data[name][i:i + BATCH_SIZE // 2] for name in validation_data}
    embedded_data_batch = workflow.approximator.summarize(batch)
    embedded_data.append(embedded_data_batch)
embedded_data = np.vstack(embedded_data)
target_params = np.concatenate([validation_data[name] for name in param_names], axis=-1)
targets = np.concatenate((target_params, embedded_data), axis=-1)
posterior_samples_test = workflow.sample(conditions=validation_data, num_samples=1,
                                         batch_size=BATCH_SIZE // 2)
posterior_samples_test = np.concatenate([posterior_samples_test[k][:, 0]  for k in param_names], axis=-1)
estimates = np.concatenate((posterior_samples_test, embedded_data), axis=-1)
estimates_mean = np.mean(estimates, axis=0)
estimates_std = np.std(estimates, axis=0)
estimates = (estimates - estimates_mean) / estimates_std
targets = (targets - estimates_mean) / estimates_std

#%%
logging.info('Train C2ST classifier')
c2st_results = bf.diagnostics.metrics.classifier_two_sample_test(
    estimates=estimates,
    targets=targets,
    return_metric_only=False,
    batch_size=BATCH_SIZE,
    standardize=False
)
logging.info(f'C2ST Accuracy: {c2st_results["score"]}')

# Coupling Flow: 0.66
# Flow Matching: 0.69
# Diffusion Model: 0.65
# Consistency Model: 0.56 (full: 0.57)

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


# %%
# read files
logging.info('Load Framingham data...')
framingham_file_names = [BASE / 'data' / f'epoch{i+1}_CV.csv' for i in range(4)]
dfs = []
for i, f in enumerate(framingham_file_names):
    new_df = pd.read_csv(f)
    new_df['epoch'] = f'epoch{i+1}'
    new_df['replicate'] = 1
    new_df['scheme'] = 'CensVisit'
    new_df['age_raw'] = new_df['age']
    new_df['age'] = new_df['ageCentered']
    dfs.append(new_df)
df_real = pd.concat(dfs, ignore_index=True)
real_data_dict = extract_batches_to_dict(df_real)

if os.path.exists(BASE / 'models' / f'posterior_samples_{network_name}.pkl'):
    logging.info('Loading posterior samples...')
    with open(BASE / 'models' / f'posterior_samples_{network_name}.pkl', 'rb') as f:
        real_posterior_samples = pickle.load(f)
    try:
        with open(BASE / 'models' / f'posterior_samples_{network_name}_full.pkl', 'rb') as f:
            real_posterior_samples_full = pickle.load(f)
    except FileNotFoundError:
        real_posterior_samples_full = {}
else:
    logging.info('Sample posterior for Framingham data...')
    real_posterior_samples = workflow.sample(conditions=real_data_dict, num_samples=1000)
    with open(BASE / 'models' / f'posterior_samples_{network_name}.pkl', 'wb') as f:
        pickle.dump(real_posterior_samples, f)

plot_data_summaries(df_real, save_path=BASE / 'plots' / 'transition_sankey.svg')


#%%
logging.info('Summarize posterior samples...')
posterior_summary = {}
for k, vals in real_posterior_samples.items():
    posterior_summary[k] = {
        "median": np.median(vals, axis=1).flatten(),
        "low": np.quantile(vals, 0.025, axis=1).flatten(),
        "high": np.quantile(vals, 0.975, axis=1).flatten(),
    }
if len(real_posterior_samples_full) > 0:
    posterior_summary_full = {}
    for k, vals in real_posterior_samples_full.items():
        posterior_summary_full[k] = {
            "median": np.median(vals, axis=1).flatten(),
            "low": np.quantile(vals, 0.025, axis=1).flatten(),
            "high": np.quantile(vals, 0.975, axis=1).flatten(),
        }

prior_samples = {}
n_prior_samples = 1000
for k, v in priors.items():
    if "alpha" in v and "beta" in v:
        # Gamma with rate parameterization
        prior_samples[k] = np.random.gamma(shape=v["alpha"], scale=1.0/v["beta"], size=n_prior_samples)
    elif "mean" in v and "sd" in v:
        prior_samples[k] = np.random.normal(loc=v["mean"], scale=v["sd"], size=n_prior_samples)
    else:
        raise ValueError(f"Unrecognized prior entry: {k}")

prior_summary = {}
for k, vals in prior_samples.items():
    prior_summary[k] = {
        "median": np.median(vals),
        "low": np.quantile(vals, 0.025),
        "high": np.quantile(vals, 0.975),
    }

# for i in range(len(framingham_file_names)):
#     logging.info(f'Epoch {i+1}')
#     for k, v in real_posterior_samples.items():
#         logging.info(f'{k} Median: {np.median(v, axis=1)[i].item()}, '
#                  f'Quantiles: {np.quantile(v, axis=1, q=[0.025, 0.975])[:, i].flatten()}')


logging.info('Plot results...')
plot_params(
    baseline,
    prior_samples,
    prior_summary,
    real_posterior_samples,
    posterior_summary,
    network_name,
    posterior_samples_2=real_posterior_samples_full,
    posterior_summary_2=posterior_summary_full,
    network_name_2=network_name+'_full',
    show_cox=True,
    idm_model='spl',
    save_path=BASE / 'plots',
)
plot_cumhaz(
    baseline,
    real_posterior_samples,
    df_real,
    network_name,
    posterior_samples_2=real_posterior_samples_full,
    network_name_2=network_name+'_full',
    #prior_samples=prior_samples,
    show_cox=False,
    save_path=BASE / 'plots',
)

#%% apply C2ST
c2st_result_real = []
c2st_result_real_random = []
embedded_real_data = workflow.approximator.summarize(real_data_dict)
embedded_real_data = np.repeat(embedded_real_data[:, None], repeats=1000, axis=1)
for i in range(len(framingham_file_names)):
    logging.info(f'Epoch {i+1} C2ST evaluation...')

    posterior_samples_test = np.concatenate([real_posterior_samples[k][i] for k in param_names], axis=-1)
    estimates_real = np.concatenate((posterior_samples_test, embedded_real_data[i]), axis=-1)
    estimates_real = (estimates_real - estimates_mean) / estimates_std
    scores = np.array([c.predict(estimates_real).flatten() for c in c2st_results['classifiers']])
    scores = np.maximum(scores, 1 - scores)
    c2st_score = np.mean(scores, axis=0)
    test_statistic = np.mean((c2st_score - 0.5) ** 2)
    c2st_result_real.append(c2st_score)
    logging.info(f'C2ST Accuracy: {np.mean(c2st_result_real[-1])}')

    # apply random classifiers
    scores_random = np.array([c['classifiers'][0].predict(estimates_real).flatten() for c in c2st_results_random])
    scores_random = np.maximum(scores_random, 1 - scores_random)
    test_statistic_random = np.mean((scores_random - 0.5)**2, axis=-1)
    p_val = np.mean(test_statistic_random > test_statistic)
    c2st_result_real_random.append((test_statistic, p_val))
    logging.info(f'C2ST Statistic: {test_statistic}, p-value: {p_val}')

bins = 20
norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "Reds_trunc",
    plt.cm.Reds(np.linspace(0.1, 1.0, 256))
)

# plot only a01
p_name = 'a01'
p_name_nice = r'$a_{01}$'
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 2), layout='constrained')
for epoch_idx in range(4):
    # compute bin assignment
    x = real_posterior_samples[p_name][epoch_idx].flatten()
    counts, bin_edges = np.histogram(x, bins=bins, density=True)
    bin_idx = np.digitize(x, bin_edges) - 1

    # compute mean color per bin
    bin_color = np.array([
        np.mean(c2st_result_real[epoch_idx][bin_idx == i]) if np.any(bin_idx == i) else 0
        for i in range(bins)
    ])

    # plot histogram manually
    for i in range(bins):
        ax[epoch_idx].bar(
            bin_edges[i],
            counts[i],
            width=bin_edges[i + 1] - bin_edges[i],
            align="edge",
            color=cmap(norm(bin_color[i])),
            edgecolor=cmap(norm(bin_color[i]))
        )

    if epoch_idx == 0:
        ax[epoch_idx].set_ylabel(f"Density of {param_names_pretty[0]}", fontsize=11)
    ax[epoch_idx].set_title(rf"Epoch {epoch_idx+1}", fontsize=11)
    m_score = np.mean(c2st_result_real[epoch_idx])
    ax[epoch_idx].text(
        0.95, 0.95,
        f"Mean C2ST={m_score:.2f}\np-value={c2st_result_real_random[epoch_idx][1]:.1f}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax[epoch_idx].transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="white", alpha=0.75)
    )
    # remove top and right spines
    ax[epoch_idx].spines['top'].set_visible(False)
    ax[epoch_idx].spines['right'].set_visible(False)
    ax[epoch_idx].set_xlim(0, prior_summary[p_name]['high'])
    ax[epoch_idx].set_xlabel(p_name_nice, fontsize=11)

# add colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), label="C2ST score\n(mean per bin)", fraction=0.02)
fig.savefig(BASE / 'plots' / f'{network_name}_real_c2st_histograms_{p_name}.pdf', bbox_inches='tight')
plt.show()


logging.info('Done.')

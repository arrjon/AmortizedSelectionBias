# %%
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras
import bayesflow as bf

#from visit_censoring.cens_visit_simulate import simulate_all_epochs
from visit_censoring.cens_visit_plotting import plot_hazard, plot_a1, plot_cumhaz
from visit_censoring.cens_visit_helper import extract_batches_to_dict, compute_gamma_params
simulate_all_epochs = lambda: 0


BASE = Path(__file__).resolve().parent
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 4))
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))

# %%
epochs = ["epoch1", "epoch2", "epoch3", "epoch4"]
params_beta = ['beta01_age', 'beta02_age', 'beta12_age', 'beta01_sex', 'beta02_sex', 'beta12_sex']
data_names = ['illt', 'ills', 'dt', 'ds', 'sex', 'age']
param_names_pretty = [r'$a_{01}$', r'$a_{02}$', r'$a_{12}$',
                      r'$s_{01}$', r'$s_{02}$', r'$s_{12}$',
                      r'$\beta_{01}^\text{sex}$', r'$\beta_{02}^\text{sex}$', r'$\beta_{12}^\text{sex}$',
                      r'$\beta_{01}^\text{age}$', r'$\beta_{02}^\text{age}$', r'$\beta_{12}^\text{age}$']
# %%
# load baseline results
with open(BASE / 'baseline.json') as f:
    _baseline = json.load(f)

naive_estimates = {
    'epoch1': {
        'a01': 0.0015/50,
        'a02': 0.0015/50,
        'a12': 0.04/50
    },
    'epoch2': {
        'a01': 0.002/50,
        'a02': 0.0002/50,
        'a12': 0.05/50
    },
    'epoch3': {
        'a01': 0.002/50,
        'a02': 0.0002/50,
        'a12': 0.04/50
    },
    'epoch4': {
        'a01': 0.002/50,
        'a02': 0.0002/50,
        'a12': 0.04/50
    }
}

baseline = {e: {} for e in epochs}
for e in epochs:
    baseline[e] = _baseline[e]
    b = _baseline[f'coefs_{e}']
    for p in params_beta:
        for k in b:
            if (k['_row'] == f"{p[4:].split('_')[1]}_{p[4:].split('_')[0]}" or
                k['_row'] == f"{p[4:].split('_')[1]}Centered_{p[4:].split('_')[0]}"):
                baseline[e][p] = {
                    'naive_cox': k['naive_cox'],
                    'weibull': k['idm_weib'],
                    'splines': k['idm_splines']
                }

    baseline[e].update({
        'a01': {'naive_cox': naive_estimates[e]['a01']},
        'a02': {'naive_cox': naive_estimates[e]['a02']},
        'a12': {'naive_cox': naive_estimates[e]['a12']}
    })
# %%
# compute mean and sd over all epochs
beta = {}
a = {}
for epoch in epochs:
    for key in ['beta01_age', 'beta02_age', 'beta12_age', 'beta01_sex', 'beta02_sex', 'beta12_sex']:
        age = baseline[epoch][key]['naive_cox']
        if key not in beta:
            beta[key] = []
        beta[key].append(age)
    for key in ['a01', 'a02', 'a12']:
        if key not in a:
            a[key] = []
        a[key].append(baseline[epoch][key]['naive_cox'])
for key in beta.keys():
    beta[key] = np.mean(beta[key]), np.std(beta[key])
for key in a.keys():
    a[key] = np.mean(a[key]), np.std(a[key])

# Framingham epoch priors
cv = 1.0
std_min = 1.0
shape_mean = 1.0  # exponential as baseline
shape_cv = 0.25
shape_params = compute_gamma_params(shape_mean, cv=shape_cv)
priors = {
    'a01': compute_gamma_params(np.mean([a['a01'][0], a['a02'][0], a['a12'][0]]), cv=cv),
    'a02': compute_gamma_params(np.mean([a['a01'][0], a['a02'][0], a['a12'][0]]), cv=cv),
    'a12': compute_gamma_params(np.mean([a['a01'][0], a['a02'][0], a['a12'][0]]), cv=cv),
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
val_data_path = BASE / 'models' / 'validation_data.pkl'
train_data_path = BASE / 'models' / 'train_data.pkl'
n_val_data = 1000
n_train_data = 20000
create_data = False
training_data, validation_data = None, None

if create_data:
    # Simulate and extract structured arrays
    print('Generate validation data...')
    df_valid = simulate_all_epochs(priors=priors, n_replicates=n_val_data // 4, ncores=n_cpus)
    validation_data = extract_batches_to_dict(df_valid, scheme='CensVisit')
    with open(val_data_path, 'wb') as f:
       pickle.dump(validation_data, f)

    print('Generate training data...')
    for i in range(10):
        print(f'  Batch {i+1}/10')
        df = simulate_all_epochs(priors=priors, n_replicates=n_train_data // 4 // 10, ncores=n_cpus)
        training_data = extract_batches_to_dict(df, scheme='CensVisit')
        with open(BASE / 'models' / f'train_data_{i}.pkl', 'wb') as f:
            pickle.dump(training_data, f)
else:
    try:
        with open(val_data_path, 'rb') as f:
            validation_data = pickle.load(f)

        if os.path.exists(train_data_path):
            with open(train_data_path, 'rb') as f:
                training_data = pickle.load(f)
        else:
            print('load single files and combine into one training data file...')
            training_data = {k: [] for k in validation_data}
            for i in range(10):
                with open(BASE / 'models' / f'train_data_{i}.pkl', 'rb') as f:
                    file = pickle.load(f)
                for k in file:
                    training_data[k].append(file[k])
            for k in training_data:
                training_data[k] = np.vstack(training_data[k])
            with open(train_data_path, 'wb') as f:
                pickle.dump(training_data, f)

    except FileNotFoundError:
        print('No data loaded.')

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
    .standardize('age', mean=-1.6, std=1)
    .log('age')
    .expand_dims(data_names, axis=-1)  # expand patients
    .concatenate(data_names + ['epoch'], into="summary_variables", axis=2)
    # parameters
    .constrain(['a01', 'a02', 'a12'], lower=0, upper=0.01)
    .standardize(['a01', 'a02', 'a12'], mean=-6, std=2)
    .log(['shape01', 'shape02', 'shape12'])
    .concatenate(param_names, into="inference_variables")
    .keep(["inference_variables", "summary_variables"])
)

# %%

if job_array_id == 0:
    BATCH_SIZE = 128
    EPOCHS = 1000
    summary_network = bf.networks.DeepSet(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))
    network_name = 'FlowMatching_DeepSet'
elif job_array_id == 1:
    BATCH_SIZE = 128
    EPOCHS = 1000
    summary_network = bf.networks.DeepSet(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.DiffusionModel(subnet_kwargs=dict(dropout=0.1))
    network_name = 'DiffusionModel_DeepSet'
elif job_array_id == 2:
    BATCH_SIZE = 128
    EPOCHS = 1000
    summary_network = bf.networks.DeepSet(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.ConsistencyModel(total_steps=EPOCHS * (n_train_data // BATCH_SIZE),
                                                     subnet_kwargs=dict(dropout=0.1))
    network_name = 'ConsistencyModel_DeepSet'
elif job_array_id == 3:
    BATCH_SIZE = 128
    EPOCHS = 100
    summary_network = bf.networks.DeepSet(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.CouplingFlow(depth=7, transform='spline',
                                                 subnet_kwargs=dict(dropout=0.1))
    network_name = 'CouplingFlow_DeepSet'
elif job_array_id == 4:
    BATCH_SIZE = 64
    EPOCHS = 1000
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))
    network_name = 'FlowMatching_SetTransformer'
elif job_array_id == 5:
    BATCH_SIZE = 64
    EPOCHS = 1000
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.DiffusionModel(subnet_kwargs=dict(dropout=0.1))
    network_name = 'DiffusionModel_SetTransformer'
elif job_array_id == 6:
    BATCH_SIZE = 64
    EPOCHS = 1000
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.ConsistencyModel(total_steps=EPOCHS * (n_train_data // BATCH_SIZE),
                                                     subnet_kwargs=dict(dropout=0.1))
    network_name = 'ConsistencyModel_SetTransformer'
elif job_array_id == 7:
    BATCH_SIZE = 64
    EPOCHS = 100
    summary_network = bf.networks.SetTransformer(summary_dim=len(param_names) * 2)
    inference_network = bf.networks.CouplingFlow(depth=7, transform='spline',
                                                 subnet_kwargs=dict(dropout=0.1))
    network_name = 'CouplingFlow_SetTransformer'
else:
    raise ValueError("Invalid job_array_id.")
model_path = BASE / 'models' / f'weibull_npe_model_{network_name}.keras'
print(job_array_id, model_path)

workflow = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=summary_network,
    inference_network=inference_network,
)
# %%
if not os.path.exists(model_path):
    history = workflow.fit_offline(
        data=training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=validation_data,
    )
    workflow.approximator.save(filepath=model_path)
else:
    history = None
    workflow.approximator = keras.saving.load_model(filepath=model_path)
# %%
if history is not None:
    fig = bf.diagnostics.loss(history)
    fig.savefig(BASE / 'plots' / f'{network_name}_training_loss.png')

# %%
if validation_data is not None:
    posterior_samples_valid = workflow.sample(conditions=validation_data, num_samples=1000,
                                              batch_size=BATCH_SIZE)
    ps_valid_adapted = adapter.forward(posterior_samples_valid, strict=False)
    valid_adapted = adapter.forward(validation_data, strict=False)

    fig = bf.diagnostics.recovery(posterior_samples_valid, validation_data, variable_names=param_names_pretty)
    fig.savefig(BASE / 'plots' / f'{network_name}_recovery.png')
    plt.show()
    fig = bf.diagnostics.recovery(ps_valid_adapted, valid_adapted, variable_names=param_names_pretty)
    fig.savefig(BASE / 'plots' / f'{network_name}_recovery_adapted.png')
    plt.show()

    fig = bf.diagnostics.calibration_ecdf(posterior_samples_valid, validation_data, variable_names=param_names_pretty,
                                          difference=True)
    ax = fig.get_axes()
    for a in ax:
        a.set_ylim(-0.25, 0.25)
    fig.savefig(BASE / 'plots' / f'{network_name}_ecdf.png')
    plt.show()
# %%
# read files
framingham_file_names = [BASE / 'data' / f'epoch{i+1}_CV.csv' for i in range(4)]
dfs = []
for i, f in enumerate(framingham_file_names):
    new_df = pd.read_csv(f)
    new_df['epoch'] = f'epoch{i+1}'
    new_df['replicate'] = 1
    new_df['scheme'] = 'CensVisit'
    new_df['age_raw'] = new_df['age']
    new_df['age'] = (new_df['age'] - np.mean(new_df['age'])) / np.std(new_df['age'])
    dfs.append(new_df)
df_real = pd.concat(dfs, ignore_index=True)
real_data_dict = extract_batches_to_dict(df_real)

if os.path.exists(BASE / 'models' / f'posterior_samples_{network_name}.pkl'):
    with open(BASE / 'models' / f'posterior_samples_{network_name}.pkl', 'rb') as f:
        real_posterior_samples = pickle.load(f)
else:
    #raise ValueError
    real_posterior_samples = workflow.sample(conditions=real_data_dict, num_samples=1000)
    with open(BASE / 'models' / f'posterior_samples_{network_name}.pkl', 'wb') as f:
        pickle.dump(real_posterior_samples, f)

# make beta comparable to other setting without standardization
std_age = np.std(df_real['age_raw'])
real_posterior_samples['beta01_age'] = real_posterior_samples['beta01_age'] / std_age
real_posterior_samples['beta02_age'] = real_posterior_samples['beta02_age'] / std_age
real_posterior_samples['beta12_age'] = real_posterior_samples['beta12_age'] / std_age

#%%
posterior_summary = {}
for k, vals in real_posterior_samples.items():
    posterior_summary[k] = {
        "median": np.median(vals, axis=1).flatten(),
        "low": np.quantile(vals, 0.025, axis=1).flatten(),
        "high": np.quantile(vals, 0.975, axis=1).flatten(),
    }

prior_samples = {}
n_samples = 1000
for k, v in priors.items():
    if "alpha" in v and "beta" in v:
        # Gamma with rate parameterization
        prior_samples[k] = np.random.gamma(shape=v["alpha"], scale=1.0/v["beta"], size=n_samples)
    elif "mean" in v and "sd" in v:
        prior_samples[k] = np.random.normal(loc=v["mean"], scale=v["sd"], size=n_samples)
    else:
        raise ValueError(f"Unrecognized prior entry: {k}")

prior_summary = {}
for k, vals in prior_samples.items():
    prior_summary[k] = {
        "median": np.median(vals),
        "low": np.quantile(vals, 0.025),
        "high": np.quantile(vals, 0.975),
    }
#%%
for i in range(len(framingham_file_names)):
    print(f'Epoch {i+1}')
    for k, v in real_posterior_samples.items():
        print(k, f'Median: {np.median(v, axis=1)[i].item()}, '
                 f'Quantiles: {np.quantile(v, axis=1, q=[0.025, 0.975])[:, i].flatten()}')
    print('\n')

#%%
plot_hazard(baseline, prior_samples, prior_summary, real_posterior_samples, posterior_summary, network_name,
            save_path=BASE / 'plots')
plot_a1(baseline, prior_samples, real_posterior_samples, network_name, save_path=BASE / 'plots')
plot_cumhaz(baseline, real_posterior_samples, df_real, network_name, save_path=BASE / 'plots')
plt.show()

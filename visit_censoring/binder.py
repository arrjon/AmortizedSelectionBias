#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
import json

#import rpy2.robjects as ro
#from rpy2.robjects import pandas2ri
#from rpy2.robjects import ListVector

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
import bayesflow as bf
#%%
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
MAX_INDV = 2299

SCHEME_NAME = {
    'Full': 0,
    'CensVisit': 1,
    'CensDeath': 2
}

epochs = ["epoch1", "epoch2", "epoch3", "epoch4"]
params_beta = ['beta01_age', 'beta02_age', 'beta12_age', 'beta01_sex', 'beta02_sex', 'beta12_sex']
params_a = ['a01', 'a02', 'a12']
params_shape = ['shape01', 'shape02', 'shape12']
data_names = ['illt', 'ills', 'dt', 'ds', 'sex', 'age']
param_names_pretty = [r'$a_{01}$', r'$a_{02}$', r'$a_{12}$',
                      r'$s_{01}$', r'$s_{02}$', r'$s_{12}$',
                      r'$\beta_{01}^\text{sex}$', r'$\beta_{02}^\text{sex}$', r'$\beta_{12}^\text{sex}$',
                      r'$\beta_{01}^\text{age}$', r'$\beta_{02}^\text{age}$', r'$\beta_{12}^\text{age}$']
#%%
def dict_to_r_list(d):
    """Recursively convert nested Python dict to R list"""
    raise NotImplementedError

def clean_dict_keys(d):
    """Rename keys with dots to underscores."""
    rename = {}
    for k in list(d.keys()):
        if '.' in k and 'beta' in k:
            new_k = re.sub(r'\.', '_', k)
            rename[k] = new_k
    d.update({rename[k]: d.pop(k) for k in rename})
    return d

def extract_batches_to_dict(df, scheme=None, epoch=None, sample_scheme_per_rep_epoch=False, seed=None):
    """
    Dict of numpy arrays shaped (n_batches, n_features).
    Batch is unique (replicate, scheme, epoch).
    Feature is per id slot within a batch.
    Patient fields are padded with zeros to MAX_INDV.
    Params are shaped (n_batches, 1).
    """
    if seed is not None:
        np.random.seed(seed)
    d = df.copy()
    if scheme is not None:
        d = d.loc[d['scheme'] == scheme]
    if epoch is not None:
        d = d.loc[d['epoch'] == epoch]
    if d.empty:
        raise ValueError("No data after filtering")

    if scheme is None and sample_scheme_per_rep_epoch:
        if d.empty:
            raise ValueError("No data to sample from")
        chosen = (
            d[["replicate", "epoch", "scheme"]]
            .drop_duplicates()
            .groupby(["replicate", "epoch"], sort=False)["scheme"]
            .apply(lambda s: np.random.choice(s.to_numpy()))
            .reset_index()
        )
        d = d.merge(chosen, on=["replicate", "epoch", "scheme"], how="inner")

    d = d.sort_values(['replicate', 'scheme', 'epoch', 'id']).reset_index(drop=True)

    patient_cols = ['id', 'illt', 'ills', 'dt', 'ds', 'sex', 'age', 'age_raw']
    param_cols = [
        'a01', 'a02', 'a12',
        'shape01', 'shape02', 'shape12',
        'beta01.sex', 'beta02.sex', 'beta12.sex',
        'beta01.age', 'beta02.age', 'beta12.age'
    ]
    meta_cols_as_features = ['replicate', 'scheme', 'epoch']

    gb = d.groupby(['replicate', 'scheme', 'epoch'], sort=True, group_keys=False)

    # check one row per id in each batch
    sizes = gb.size().to_numpy()
    ids_unique = gb['id'].nunique().to_numpy()
    if not np.all(sizes == ids_unique):
        raise ValueError("Each batch must have at most one row per id")

    batch_keys = list(gb.groups.keys())
    n_batches = len(batch_keys)

    # MAX_INDV
    if MAX_INDV <= 0:
        raise ValueError("MAX_INDV must be positive")

    def pad_array(values, length):
        out = np.zeros(length, dtype=np.float32)
        n = len(values)
        out[:n] = np.asarray(values, dtype=np.float32)
        return out

    def stack_patient(col):
        if col not in d.columns:
            return None
        mats = []
        for key in batch_keys:
            g = gb.get_group(key).sort_values('id')
            mats.append(pad_array(g[col].to_numpy(), MAX_INDV))
        return np.vstack(mats)  # (n_batches, MAX_INDV)

    def stack_meta(col):
        mats = []
        for key in batch_keys:
            rep, sch, ep = key
            if col == 'replicate':
                val = int(rep)
            elif col == 'scheme':
                val = SCHEME_NAME.get(sch, -1)
            elif col == 'epoch':
                val = float(ep[-1])
            else:
                continue

            arr = np.zeros(MAX_INDV, dtype=np.float32)
            n = gb.get_group(key)['id'].nunique()
            arr[:n] = val
            mats.append(arr)
        return np.vstack(mats)  # (n_batches, MAX_INDV)

    def stack_param(col):
        if col not in d.columns:
            return None
        vals = []
        for key in batch_keys:
            g = gb.get_group(key)
            vals.append(g[col].iloc[0])
        arr = np.asarray(vals, dtype=np.float32)
        return arr.reshape(n_batches, 1)

    result = {}

    # patient level arrays
    for col in patient_cols:
        arr = stack_patient(col)
        if arr is not None:
            result[col] = arr

    # meta as features with zero padding
    for col in meta_cols_as_features:
        result[col] = stack_meta(col)

    # params per batch
    for col in param_cols:
        arr = stack_param(col)
        if arr is not None:
            result[col] = arr
    return clean_dict_keys(result)

def compute_gamma_params(mean, cv):
    """
    Compute gamma distribution parameters from mean and coefficient of variation.

    Parameters:
    -----------
    mean : float
        Desired mean of the gamma distribution
    cv : float
        Coefficient of variation (SD/mean), default=0.3 for moderate uncertainty

    Returns:
    --------
    dict with 'alpha' (shape) and 'beta' (rate) parameters
    """
    alpha = 1 / (cv ** 2)
    beta = alpha / mean
    return {'alpha': alpha, 'beta': beta}


def simulate_epoch(priors_epoch, n_replicates, ncores):
    raise NotImplementedError

# Run simulation for all epochs
def simulate_all_epochs(priors, n_replicates, ncores=1, epochs=None):
    raise NotImplementedError

#%%
# load baseline results
with open('baseline.json') as f:
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
baseline['epoch1'].keys()
#%%
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


#%%
val_data_path = 'models/binder_validation_data.pkl'
train_data_path = 'models/binder_train_data.pkl'
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
        with open(f'binder_train_data_{i}.pkl', 'wb') as f:
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
                with open(f'binder_train_data_{i}.pkl', 'rb') as f:
                    file = pickle.load(f)
                for k in file:
                    training_data[k].append(file[k])
            for k in training_data:
                training_data[k] = np.vstack(training_data[k])
            with open(train_data_path, 'wb') as f:
                pickle.dump(training_data, f)

    except FileNotFoundError:
        print('No data loaded.')

if validation_data is not None:
    validation_data['epoch'] = (validation_data['epoch'] -1).astype(int)
if training_data is not None:
    training_data['epoch'] = (training_data['epoch'] -1).astype(int)

adapter = (
    bf.adapters.Adapter()
    .to_array()
    .one_hot('epoch', num_classes=4)
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    # observables
    .constrain(['illt', 'dt'], lower=0, upper=1826, inclusive='both')
    .standardize(['illt', 'dt'], mean=-25, std=20)
    .standardize('age', mean=-1.6, std=1)
    .log('age')
    .expand_dims(data_names, axis=-1)  # expand patients
    .concatenate(data_names + ['epoch'], into="summary_variables", axis=2)
    # parameters
    .constrain(['a01', 'a02', 'a12'], lower=0, upper=1)
    .standardize(['a01', 'a02', 'a12'], mean=-10, std=2)
    .log(['shape01', 'shape02', 'shape12'])
    .concatenate(param_names, into="inference_variables")
    .keep(["inference_variables", "summary_variables"])
)

#%%
BATCH_SIZE = 64

summary_network = bf.networks.DeepSet(summary_dim=len(param_names) * 2, dropout=0.1)
if job_array_id == 0:
    EPOCHS = 1000
    inference_network = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1),
                                                 integrate_kwargs=dict(method='rk45', steps='adaptive'))
    network_name = 'FlowMatching_DeepSet'
elif job_array_id == 1:
    EPOCHS = 1000
    inference_network = bf.networks.DiffusionModel(subnet_kwargs=dict(dropout=0.1),
                                                   integrate_kwargs=dict(method='euler_maruyama', steps=200))
    network_name = 'DiffusionModel_DeepSet'
elif job_array_id == 2:
    EPOCHS = 1000
    inference_network = bf.networks.ConsistencyModel(total_steps=EPOCHS * (n_train_data // BATCH_SIZE),
                                                     subnet_kwargs=dict(dropout=0.1))
    network_name = 'ConsistencyModel_DeepSet'
elif job_array_id == 3:
    EPOCHS = 100
    inference_network = bf.networks.CouplingFlow(depth=7, transform='spline',
                                                 subnet_kwargs=dict(dropout=0.1))
    network_name = 'CouplingFlow_DeepSet'
else:
    raise ValueError("Invalid job_array_id.")
model_path = f'models/binder_weibull_npe_model_{network_name}.keras'
print(job_array_id, model_path)

workflow = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=summary_network,
    inference_network=inference_network,
)
#%%
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
#%%
if history is not None:
    fig = bf.diagnostics.loss(history)
    fig.savefig(f'plots/binder/binder_{network_name}_training_loss.png')
#%%
def sample_in_batches(data, num_samples,):
    # sample in batches
    posterior_samples = None
    for i in range(0, len(data['id']), BATCH_SIZE):
        batch_data = {k: v[i:i + BATCH_SIZE] for k, v in data.items()}
        batch_samples = workflow.sample(conditions=batch_data, num_samples=num_samples)
        if i == 0:
            posterior_samples = batch_samples
        else:
            for key in posterior_samples.keys():
                posterior_samples[key] = np.vstack([posterior_samples[key], batch_samples[key]])
    return posterior_samples
#%%
if validation_data is not None:
    posterior_samples_valid = sample_in_batches(validation_data, num_samples=1000)
    ps_valid_adapted = adapter.forward(posterior_samples_valid, strict=False)
    valid_adapted = adapter.forward(validation_data, strict=False)

    fig = bf.diagnostics.recovery(posterior_samples_valid, validation_data, variable_names=param_names_pretty)
    fig.savefig(f'plots/binder/binder_{network_name}_recovery.png')
    plt.show()

    fig = bf.diagnostics.calibration_ecdf(posterior_samples_valid, validation_data, variable_names=param_names_pretty,
                                          difference=True)
    ax = fig.get_axes()
    for a in ax:
        a.set_ylim(-0.25, 0.25)
    fig.savefig(f'plots/binder/binder_{network_name}_ecdf.png')
    plt.show()

    fig = bf.diagnostics.calibration_ecdf(ps_valid_adapted, valid_adapted, variable_names=param_names_pretty,
                                          difference=True)
    ax = fig.get_axes()
    for a in ax:
        a.set_ylim(-0.25, 0.25)
    fig.savefig(f'plots/binder/binder_{network_name}_ecdf_adapted.png')
    plt.show()
#%%
# read files
framingham_file_names = [f'epoch{i+1}_CV.csv' for i in range(4)]
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
real_data_dict['epoch'] = (real_data_dict['epoch'] -1).astype(int)

#%%
real_posterior_samples = workflow.sample(conditions=real_data_dict, num_samples=1000)
with open(f'models/posterior_samples_binder_{network_name}.pkl', 'wb') as f:
    pickle.dump(real_posterior_samples, f)

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

from binder_plotting import plot_hazard, plot_a1, plot_a1_median, plot_cumhaz

plot_hazard(baseline, prior_samples, prior_summary, real_posterior_samples, posterior_summary, network_name)
plot_a1(baseline, prior_samples, real_posterior_samples, network_name)
#plot_a1_median(baseline, prior_samples, real_posterior_samples, network_name)
plot_cumhaz(baseline, real_posterior_samples, df_real, network_name)

print('Done!')

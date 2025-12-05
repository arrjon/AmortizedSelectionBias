import numpy as np
import pandas as pd
import re

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import ListVector


ro.r.source('illness_death_simulation_weibull.R')
simulate_from_priors_df = ro.globalenv['simulate_from_priors_df']


def dict_to_r_list(d):
    """Recursively convert nested Python dict to R list"""
    r_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            r_dict[key] = dict_to_r_list(value)
        elif isinstance(value, (int, float)):
            r_dict[key] = ro.FloatVector([value])
        else:
            r_dict[key] = value
    return ListVector(r_dict)


def clean_dict_keys(d):
    """Rename keys with dots to underscores."""
    rename = {}
    for k in list(d.keys()):
        if '.' in k and 'beta' in k:
            new_k = re.sub(r'\.', '_', k)
            rename[k] = new_k
    d.update({rename[k]: d.pop(k) for k in rename})
    return d


def extract_batches_to_dict(df, scheme=None, epoch=None, sample_scheme_per_rep_epoch=False, seed=None, max_indiv=2299):
    """
    Dict of numpy arrays shaped (n_batches, n_features).
    Batch is unique (replicate, scheme, epoch).
    Feature is per id slot within a batch.
    Patient fields are padded with zeros to max_indiv.
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

    scheme_name = {
        'Full': 0,
        'CensVisit': 1,
        'CensDeath': 2
    }

    if max_indiv <= 0:
        raise ValueError("max_indiv must be positive")

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
            mats.append(pad_array(g[col].to_numpy(), max_indiv))
        return np.vstack(mats)  # (n_batches, max_indiv)

    def stack_meta(col):
        mats = []
        for key in batch_keys:
            rep, sch, ep = key
            if col == 'replicate':
                val = int(rep)
            elif col == 'scheme':
                val = scheme_name.get(sch, -1)
            elif col == 'epoch':
                val = float(ep[-1])
            else:
                continue

            arr = np.zeros(max_indiv, dtype=np.float32)
            n = gb.get_group(key)['id'].nunique()
            arr[:n] = val
            mats.append(arr)
        return np.vstack(mats)  # (n_batches, max_indiv)

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
    """
    Simulate illness-death data for a specific epoch.

    Parameters:
    -----------
    priors_epoch : dict
        Prior of parameters for a single epoch
    n_replicates : int
        Number of replicates
    ncores : int
        Number of cores for parallel processing

    Returns:
    --------
    pandas.DataFrame with simulated data
    """
    # Convert Python dict to R list using helper function
    r_priors = dict_to_r_list(priors_epoch)

    # Call R function
    result_single = simulate_from_priors_df(
        priors=r_priors,
        N=n_replicates,
        ncores=ncores
    )

    with (ro.default_converter + pandas2ri.converter).context():
        df_single = ro.conversion.get_conversion().rpy2py(result_single)
    return df_single

# Run simulation for all epochs
def simulate_all_epochs(priors, n_replicates, ncores=1, epochs=None):
    """
    Simulate illness-death data for multiple Framingham epochs. Each epoch sampled new parameters.

    Parameters:
    -----------
    priors : dict
        Prior of parameters for all epochs
    n_replicates : int
        Number of replicates per epoch
    ncores : int
        Number of cores for parallel processing
    epochs : list of str, optional
        List of epoch names to simulate. If None, simulate all epochs.

    Returns:
    --------
    pandas.DataFrame with simulated data from all requested epochs
    """
    if epochs is None:
        epochs = ['epoch1', 'epoch2', 'epoch3', 'epoch4']

    dfs = []
    for epoch_name in epochs:
        priors_epoch = {epoch_name: priors}
        df_single = simulate_epoch(priors_epoch, n_replicates=n_replicates, ncores=ncores)
        dfs.append(df_single)
    dfs = pd.concat(dfs, ignore_index=True)
    return dfs

def simulate_params(params, priors, ncores=1, epochs=None):
    if epochs is None:
        epochs = ['epoch1', 'epoch2', 'epoch3', 'epoch4']

    dfs = []
    for epoch_name in epochs:
        for i in range(params['a01'].shape[1]):
            params_epoch = {epoch_name: {re.sub(r'_', '.', k): v[0, i].item() for k, v in params.items()}}
            prior_epoch = {epoch_name: priors}
            # Convert Python dict to R list using helper function
            r_params = dict_to_r_list(params_epoch)
            r_prior = dict_to_r_list(prior_epoch)

            # Call R function
            result_single = simulate_from_priors_df(
                prior=r_prior,
                params=r_params,
                N=1,
                ncores=ncores
            )

            with (ro.default_converter + pandas2ri.converter).context():
                df_single = ro.conversion.get_conversion().rpy2py(result_single)

            df_single['replicate'] = i
            dfs.append(df_single)
    dfs = pd.concat(dfs, ignore_index=True)
    return dfs
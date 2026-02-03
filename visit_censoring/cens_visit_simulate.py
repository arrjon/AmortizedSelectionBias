import pandas as pd
import re

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import ListVector


ro.r.source('visit_censoring/illness_death_simulation_weibull.R')
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

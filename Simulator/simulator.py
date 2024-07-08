import numpy as np
from rpy2.robjects import r, conversion, pandas2ri

from helper_functions import normalize_household_data, dict_to_named_list

pandas2ri.activate()
r.source('Simulator/Simulator.R')
model_r = r['simulate_and_reformat']

param_names = ['alpha', 'beta', 'delta',
               'mu_inf_SC', 'mu_inf_SA', 'mu_inf_AI', 'mu_inf_AC', 'mu_inf_AA',
               'mu_susc_C', 'mu_susc_A',
               'mu_protect_acq', 'mu_protect_transm']


def simulator(params: np.ndarray,
              variant: str = "alpha",
              selection_procedure: str = "pedcov") -> np.ndarray:
    """
    Simulate data with given parameters and reformat it to a numpy array.
    :param params: parameters for the simulation
    :param variant: variant of the virus (alpha or omicron)
    :param selection_procedure: selection procedure for the households (pedcov or random)
    :return: simulated data as numpy array
    """
    # transform parameters to correct scale
    scaled_params = np.copy(params)
    scaled_params[3:] = np.exp(scaled_params[3:])
    # create dict from params and param_names
    par_dict = dict(zip(param_names, scaled_params))
    # update dict with fixed parameters
    par_dict.update({'variant': variant, 'selection_procedure': selection_procedure})

    # minimal_length should be the maximal length of the time series in the real data set
    if par_dict['variant'] == "alpha":
        minimal_length = 8
    elif par_dict['variant'] == "omicron":
        minimal_length = 9
    else:
        raise ValueError("Variant not supported. Must be 'alpha' or 'omicron'.")

    # simulate data
    sim_data_r = model_r(dict_to_named_list(par_dict))
    # convert to pandas dataframe
    sim_data_full = conversion.rpy2py(sim_data_r)
    # normalize data and return as numpy array
    sim_data_norm = normalize_household_data(sim_data_full, minimal_length=minimal_length)
    return sim_data_norm

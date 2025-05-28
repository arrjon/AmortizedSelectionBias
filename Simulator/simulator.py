import numpy as np
import pandas as pd
from rpy2.robjects import r, conversion, pandas2ri

from helper_functions import normalize_household_data, dict_to_named_list

pandas2ri.activate()
r.source('Simulator/Simulator.R')
model_r = r['simulate_and_reformat']

PARAM_NAMES = ['alpha',  # alpha is not estimated
               'beta', 'delta',
               'mu_inf_SC', 'mu_inf_SI',
               'mu_inf_AI', 'mu_inf_AC', 'mu_inf_AA',
               'mu_susc_C', 'mu_susc_I',
               'mu_protect_acq',
               'mu_protect_transm']  # last two are fixed

FIXED_PARAMETERS = {
    'alpha': {
        "mu_protect_acq": 0.8,  # not enough data to estimate
        "mu_protect_transm": 1.  # not enough data to estimate
    },
    'omicron': {
        "mu_protect_acq": 0.8,  # not enough data to estimate
        "mu_protect_transm": 0.8  # not enough data to estimate
    }
}

PROCEDURES = ['pedcov', 'random', 'original_pedcov', 'original_random', 'adult', 'sampling1', 'samplingIG']

alpha_community_infection = np.array([0.0005, 0.002, 0.003])
omicron_community_infection = alpha_community_infection * 10

def simulator(alpha, beta, delta,
              mu_inf_SC, mu_inf_SI,
              mu_inf_AI, mu_inf_AC, mu_inf_AA,
              mu_susc_C, mu_susc_I,
              mu_protect_acq, mu_protect_transm,
              selection_procedure, variant) -> dict[str, pd.DataFrame]:
    """
    Simulate data with given parameters and reformat it to a numpy array.
    """
    if variant not in ['alpha', 'omicron']:
        raise ValueError(f"Variant '{variant}' not supported. Must be 'alpha' or 'omicron'.")
    if selection_procedure not in PROCEDURES:
        raise ValueError(f"Selection procedure '{selection_procedure}' not supported. "
                         f"Must be one of {PROCEDURES}.")

    # create dict from param_names, make sure inputs are floats or strings
    par_dict = {
        'alpha': float(alpha),
        'beta': float(beta),
        'delta': float(delta),
        'mu_inf_SC': float(mu_inf_SC),
        'mu_inf_SI': float(mu_inf_SI),
        'mu_inf_AI': float(mu_inf_AI),
        'mu_inf_AC': float(mu_inf_AC),
        'mu_inf_AA': float(mu_inf_AA),
        'mu_susc_C': float(mu_susc_C),
        'mu_susc_I': float(mu_susc_I),
        'mu_protect_acq': float(mu_protect_acq),
        'mu_protect_transm': float(mu_protect_transm),
        'variant': str(variant),
        'selection_procedure': str(selection_procedure)
    }

    # simulate data
    sim_data_r = model_r(dict_to_named_list(par_dict))
    # convert to pandas dataframe
    sim_data_full = conversion.rpy2py(sim_data_r)
    # normalize data and return as numpy array
    #sim_data_norm = normalize_household_data(sim_data_full, minimal_length=minimal_length)
    return dict(sim_data_df=sim_data_full)


def simulator_both_variants(log_params: np.ndarray,
                            batchable_context: [str, [float, float]],
                            minimal_length: int = 9) -> np.ndarray:
    """
    Simulate data for both variants with given parameters and reformat it to a numpy array.
    :param log_params: parameters for the simulation
    :param batchable_context: selection procedure for the simulation (pedcov or random) and alpha value
    :param minimal_length: minimal length of the data
    :return: simulated data as numpy array
    """
    selection_procedure, params_alpha = batchable_context

    if selection_procedure not in PROCEDURES:
        raise ValueError(f"Selection procedure '{selection_procedure}' not supported. "
                         f"Must be one of {PROCEDURES}.")

    # transform parameters to correct scale
    un_scaled_params = np.copy(log_params)  # copy to avoid changing input
    # create dict from param_names, params might have different length
    p_i = 0
    data = []
    for vi, variant in enumerate(['alpha', 'omicron']):
        par_dict = {}

        for name in PARAM_NAMES:
            if name in FIXED_PARAMETERS[variant]:
                par_dict.update({name: FIXED_PARAMETERS[variant][name]})
            elif name == 'alpha':
                par_dict.update({name: float(params_alpha[vi])})
            # all parameters besides delta are log-transformed
            elif name == 'delta':
                par_dict.update({name: un_scaled_params[p_i]})
                p_i += 1
            else:
                par_dict.update({name: np.exp(un_scaled_params[p_i])})
                p_i += 1
        # update dict with fixed hyperparameters, make sure these are strings
        par_dict.update({'variant': str(variant), 'selection_procedure': str(selection_procedure)})

        # simulate data
        sim_data_r = model_r(dict_to_named_list(par_dict))
        # convert to pandas dataframe
        sim_data_full = conversion.rpy2py(sim_data_r)
        # normalize data and return as numpy array
        sim_data_norm = normalize_household_data(sim_data_full, minimal_length=minimal_length)
        # add variant as feature
        if variant == 'alpha':
            variant_feature = np.zeros((sim_data_norm.shape[0], minimal_length, 1))
        else:
            variant_feature = np.ones((sim_data_norm.shape[0], minimal_length, 1))
        sim_data_norm = np.concatenate((sim_data_norm, variant_feature), axis=-1)
        # append data
        data.append(sim_data_norm)
    return np.concatenate(data, axis=0)

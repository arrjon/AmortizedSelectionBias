import numpy as np
from rpy2.robjects import r, conversion, pandas2ri

from helper_functions import normalize_household_data, dict_to_named_list

pandas2ri.activate()
r.source('Simulator/Simulator.R')
model_r = r['simulate_and_reformat']

PARAM_NAMES = ['alpha', 'beta', 'delta',
               'mu_inf_SC', 'mu_inf_SA', 'mu_inf_AI', 'mu_inf_AC', 'mu_inf_AA',
               'mu_susc_C', 'mu_susc_A',
               'mu_protect_acq', 'mu_protect_transm']

PROCEDURES = ['pedcov', 'random', 'original_pedcov', 'original_random', 'adult', 'sampling1', 'samplingIG']


def simulator(log_params: np.ndarray,
              selection_procedure: str,
              variant: str,
              minimal_length: int = 9,
              fixed_parameters_dict: dict = None) -> np.ndarray:
    """
    Simulate data with given parameters and reformat it to a numpy array.
    :param log_params: parameters for the simulation
    :param selection_procedure: selection procedure for the simulation (pedcov or random)
    :param variant: variant of the simulation (alpha or omicron)
    :param minimal_length: minimal length of the data
    :param fixed_parameters_dict: dictionary with fixed parameters for each variant
    :return: simulated data as numpy array
    """
    if variant not in ['alpha', 'omicron']:
        raise ValueError(f"Variant '{variant}' not supported. Must be 'alpha' or 'omicron'.")
    if selection_procedure not in PROCEDURES:
        raise ValueError(f"Selection procedure '{selection_procedure}' not supported. "
                         f"Must be one of {PROCEDURES}.")

    if fixed_parameters_dict is None:
        fixed_parameters = []
    else:
        fixed_parameters = fixed_parameters_dict[variant]

    # transform parameters to correct scale
    un_scaled_params = np.copy(log_params)  # copy to avoid changing input
    # create dict from param_names, params might have different length
    par_dict = {}
    p_i = 0
    for name in PARAM_NAMES:
        if name in fixed_parameters:
            par_dict.update({name: fixed_parameters[name]})
        # all parameters besides alpha and delta are log-transformed
        elif name == 'delta' or name == 'alpha':
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
    return sim_data_norm


def simulator_both_variants(log_params: np.ndarray,
                            selection_procedure: str,
                            minimal_length: int = 9,
                            fixed_parameters_dict: dict = None) -> np.ndarray:
    """
    Simulate data for both variants with given parameters and reformat it to a numpy array.
    :param log_params: parameters for the simulation
    :param selection_procedure: selection procedure for the simulation (pedcov or random)
    :param minimal_length: minimal length of the data
    :param fixed_parameters_dict: dictionary with fixed parameters for each variant
    :return: simulated data as numpy array
    """
    if selection_procedure not in ['pedcov', 'random', 'original_pedcov', 'original_random']:
        raise ValueError(f"Selection procedure '{selection_procedure}' not supported. "
                         f"Must be 'pedcov', 'random', 'original_pedcov' or 'original_random'.")

    if fixed_parameters_dict is None:
        fixed_parameters_dict = {'alpha': [], 'omicron': []}

    # transform parameters to correct scale
    un_scaled_params = np.copy(log_params)  # copy to avoid changing input
    # create dict from param_names, params might have different length
    p_i = 0
    data = []
    for variant in ['alpha', 'omicron']:
        par_dict = {}

        for name in PARAM_NAMES:
            if name in fixed_parameters_dict[variant]:
                par_dict.update({name: fixed_parameters_dict[variant][name]})
            # all parameters besides alpha and delta are log-transformed
            elif name == 'delta' or name == 'alpha':
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

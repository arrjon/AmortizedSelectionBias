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

fixed_parameters_alpha = {
    "alpha": 0.001,
    "delta": 1.3,
    "mu_protect_acq": 0.8,
    "mu_protect_transm": 1.
}

fixed_parameters_omicron = {
    "alpha": 0.002,
    "delta": 1.4,
    "mu_protect_acq": 0.8,
    "mu_protect_transm": 0.8
}


def simulator(params: np.ndarray,
              selection_procedure: str,
              variant: str,
              minimal_length: int = 9,
              fix_parameters: bool = True) -> np.ndarray:
    """
    Simulate data with given parameters and reformat it to a numpy array.
    :param params: parameters for the simulation
    :param selection_procedure: selection procedure for the simulation (pedcov or random)
    :param variant: variant of the simulation (alpha or omicron)
    :param minimal_length: minimal length of the data
    :param fix_parameters: if True, fixed parameters are used
    :return: simulated data as numpy array
    """
    if variant == "alpha":
        fixed_parameters = fixed_parameters_alpha
    elif variant == "omicron":
        fixed_parameters = fixed_parameters_omicron
    else:
        raise ValueError(f"Variant '{variant}' not supported. Must be 'alpha' or 'omicron'.")
    if selection_procedure not in ['pedcov', 'random']:
        raise ValueError(f"Selection procedure '{selection_procedure}' not supported. Must be 'pedcov' or 'random'.")

    # transform parameters to correct scale
    un_scaled_params = np.copy(params)
    if not fix_parameters:
        fixed_parameters = None
    # create dict from param_names, params might have different length
    par_dict = {}
    p_i = 0
    for name in param_names:
        if fixed_parameters is not None and name in fixed_parameters:
            par_dict.update({name: fixed_parameters[name]})
        # all parameters starting with mu are log transformed, fixed parameters are not transformed
        elif name.startswith('mu') or name == 'beta':
        #elif name.startswith('mu'):
            par_dict.update({name: np.exp(un_scaled_params[p_i])})
            p_i += 1
        else:
            par_dict.update({name: un_scaled_params[p_i]})
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

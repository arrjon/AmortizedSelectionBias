from pathlib import Path

from PedCov.helper_functions import normalize_household_data

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from rpy2.robjects import ListVector
from rpy2.robjects.vectors import FloatVector


BASE = Path(__file__).resolve().parent

# Defining the R script and loading the instance in Python
ro.r['source'](str(BASE / 'Simulator.R'))

# Loading the function we have defined in R.
simulate_with_r_function = ro.globalenv['simulate']


def dict_to_named_r_list(d):
    r_list = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Inner dict → named FloatVector
            vec = FloatVector(list(value.values()))
            vec.names = ro.StrVector(list(value.keys()))
            r_list[key] = vec
        else:
            # Scalar → single-element FloatVector
            r_list[key] = FloatVector([value])
    return ListVector(r_list)

# -----------------------------------------------------------------------------
# Python wrapper to run across all households
# -----------------------------------------------------------------------------
class OutbreakSimulator:
    def __init__(self, variant='alpha'):
        self.variant   = variant

        if variant == "alpha":
            self.p_asympto = 0.4
            mean_incub, sd_incub = 4.42, 2.3

            self.shape_generation_time, self.scale_generation_time = 2, 1/0.44
        elif variant == "omicron":
            self.p_asympto = 0.3
            mean_incub, sd_incub = 3.09, 1.64

            self.shape_generation_time, self.scale_generation_time = 3.531, 1/1.098
        else:
            raise ValueError(f"Unknown variant: {variant}")
        self.shapeIncub = mean_incub**2 / sd_incub**2
        self.scaleIncub = sd_incub**2 / mean_incub

    def __call__(self,
                 alpha, beta, delta,
                 mu_inf_SI, mu_inf_SC, mu_inf_AI, mu_inf_AC, mu_inf_AA,
                 mu_susc_I, mu_susc_C,
                 mu_protect_acq, mu_protect_transm,
                 selection_procedure="random",
                 return_df=False
                 ):
        if selection_procedure not in ["random", "pedcov", "adultcov", "all"]:
            raise ValueError("Method must be one of: 'random', 'pedcov', 'adultcov', 'all'")

        # Prepare the parameters for the R function
        params_dict = dict(
            beta=beta,
            alpha=alpha,
            delta=delta,
            mu_inf=dict(SI=mu_inf_SI, SC=mu_inf_SC, AI=mu_inf_AI, AC=mu_inf_AC, AA=mu_inf_AA),
            mu_susc=dict(I=mu_susc_I, C=mu_susc_C),
            mu_protect=dict(acq=mu_protect_acq, transm=mu_protect_transm)
        )
        # converting it into r object
        r_params = dict_to_named_r_list(params_dict)

        # Invoking the R function and getting the result
        df_sim_r = simulate_with_r_function(r_params, str(self.variant), str(selection_procedure))

        # Converting it back to a pandas dataframe.
        with (ro.default_converter + pandas2ri.converter).context():
            df_sim = ro.conversion.get_conversion().rpy2py(df_sim_r)

        if selection_procedure == "all":
            # If 'all' is selected, we need to split the dataframe into three parts
            df_list = [
                df_sim[df_sim['select_process'] == 'pedcov'],
                df_sim[df_sim['select_process'] == 'adultcov'],
                df_sim[df_sim['select_process'] == 'random']
            ]
            sim_norm_list = []
            for df in df_list:
                sim_norm = normalize_household_data(df)
                sim_norm_list.append(sim_norm)

            if return_df:
                return {
                    'pedcov': dict(sim_data_df=df_list[0], sim_data=sim_norm_list[0]),
                    'adultcov': dict(sim_data_df=df_list[1], sim_data=sim_norm_list[1]),
                    'random': dict(sim_data_df=df_list[2], sim_data=sim_norm_list[2]),
                }
            else:
                return {
                    'pedcov': dict(sim_data=sim_norm_list[0]),
                    'adultcov': dict(sim_data=sim_norm_list[1]),
                    'random': dict(sim_data=sim_norm_list[2]),
                }

        # normalize the household PedCov for input to the neural network
        sim_norm = normalize_household_data(df_sim)
        if return_df:
            return dict(sim_data_df=df_sim, sim_data=sim_norm)
        else:
            return dict(sim_data=sim_norm)

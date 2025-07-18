from PedCov.helper_functions import normalize_household_data

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from rpy2.robjects import ListVector
from rpy2.robjects.vectors import FloatVector

# Defining the R script and loading the instance in Python
ro.r['source']('PedCov/Simulator.R')

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
        #self.n_repeat  = n_repeat
        self.minimal_length = 8  # minimal length of the household PedCov -> time steps
        #self.simple_simulation = simple_simulation  # whether to use incubation delay or not

        if variant == "alpha":
            self.p_asympto = 0.4
            mean_incub, sd_incub = 4.42, 2.3
            #self.minIncub, self.maxIncub = 2, 7
            #self.shapeIncubAsymp = 4.0  # note: similar to uniform distribution before
            #self.scaleIncubAsymp = 1.286

            self.shape_generation_time, self.scale_generation_time = 2, 1/0.44
            #self.delayDist = [6.9368753, 0.7376425, -3.0000000, 5.6516107, 0.9719026, 2.0000000]
        elif variant == "omicron":
            self.p_asympto = 0.3
            mean_incub, sd_incub = 3.09, 1.64
            #self.minIncub, self.maxIncub = 1, 5
            #self.shapeIncubAsymp = 7.0  # note: similar to uniform distribution before
            #self.scaleIncubAsymp = 2./3

            self.shape_generation_time, self.scale_generation_time = 3.531, 1/1.098
            #self.delayDist = [8.4310842, 0.9508425, -4.0000000, 3.8209443, 1.2737556, 1.0000000]
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
                sim_norm = normalize_household_data(df, minimal_length=self.minimal_length)
                sim_norm_list.append(sim_norm)

            return {
                'pedcov': dict(sim_data_df=df_list[0], sim_data=sim_norm_list[0]),
                'adultcov': dict(sim_data_df=df_list[1], sim_data=sim_norm_list[1]),
                'random': dict(sim_data_df=df_list[2], sim_data=sim_norm_list[2]),
            }

        # normalize the household PedCov for input to the neural network
        sim_norm = normalize_household_data(df_sim, minimal_length=self.minimal_length)
        if return_df:
            return dict(sim_data_df=df_sim, sim_data=sim_norm)
        else:
            return dict(sim_data=sim_norm)

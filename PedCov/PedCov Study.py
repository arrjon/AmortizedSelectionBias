#%% md
# # Simulation Based Inference to Remove Sampling Bias - Household Studies
# 
#%% md
# Household infection model with parameters `beta`, `delta`, `mu_inf_SI`, `mu_inf_SC`, `mu_inf_AI`, `mu_inf_AC`, `mu_inf_AA`, `mu_susc_I`, `mu_susc_C`.
#%%
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
batch_size = 64
variants = ['alpha', 'omicron']
# name of the scenario, used for saving files
if len(variants) == 2:
    scenario_name = 'pedcov_both_variants'
elif variants[0] == 'alpha':
    scenario_name = f'pedcov'
elif variants[0] == 'omicron':
    scenario_name = f'pedcov_omicron'
else:
    raise ValueError(f"Unknown variant: {variants}")
print('Scenario:', scenario_name)
#%% md
# ## Define model and prior
# 
# We fix some parameters such that the model becomes identifiable (checked with STAN). We fix the parameters `alpha`, `mu_protect_acq`, `mu_protect_transm`.
#%%
import pickle
import itertools
from joblib import Parallel, delayed
import numpy as np

import keras
import bayesflow as bf

from PedCov.stan import get_stan_posterior
from PedCov.simulator import OutbreakSimulator
from PedCov.helper_functions import list_of_dicts_to_dict_of_lists
#%%
param_names = {  # comment out parameters that should not be estimated
    #'alpha': r'$\alpha$',
    'beta': r'$\beta$', 'delta': r'$\delta$',
    'mu_inf_SI': r'$\mu_\text{infectiousness}^\text{symptomatic Infant}$', 'mu_inf_SC': r'$\mu_\text{infectiousness}^\text{symptomatic Child}$',
    'mu_inf_AI': r'$\mu_\text{infectiousness}^\text{asymptomatic Infant}$', 'mu_inf_AC': r'$\mu_\text{infectiousness}^\text{asymptomatic Child}$', 'mu_inf_AA': r'$\mu_\text{infectiousness}^\text{asymptomatic Adult}$',
    'mu_susc_I': r'$\mu_\text{susceptibility}^\text{Infant}$', 'mu_susc_C': r'$\mu_\text{susceptibility}^\text{Child}$',
    'mu_protect_acq': r'$\mu_\text{protect}^\text{acq}$', 'mu_protect_transm': r'$\mu_\text{protect}^\text{transm}$'
}

full_name_list = ['alpha', 'beta', 'delta',
                  'mu_inf_SI', 'mu_inf_SC', 'mu_inf_AI', 'mu_inf_AC', 'mu_inf_AA',
                  'mu_susc_I', 'mu_susc_C',
                  'mu_protect_acq', 'mu_protect_transm']
#%%
# define the prior
def meta() -> dict:
    _selection_procedure_id = np.random.choice([0, 1, 2])  # 0 for random, 1 for pedcov, 2 for adult
    _variant = np.random.choice(variants)  # 0 for alpha, 1 for omicron
    return dict(
        variant=_variant,  # alpha or omicron
        variant_id=0 if _variant == 'alpha' else 1,
        selection_procedure=['random', 'pedcov', 'adultcov'][_selection_procedure_id],
        selection_procedure_id=_selection_procedure_id
    )

def prior(variant) -> dict:
    var = 0.7 # was 1 before
    if variant == 'alpha':
        alpha_fixed = 0.001  # fixed for alpha
    elif variant == 'omicron':
        alpha_fixed = 0.01  # fixed for omicron
    else:
        raise ValueError(f"Unknown variant: {variant}")

    params = {
        'alpha': np.random.uniform(0, 0.1) if 'alpha' in param_names.keys() else alpha_fixed,
        'beta': np.random.gamma(shape=2.0, scale=1/2.0) if 'beta' in param_names.keys() else 0.3,
        'delta': np.random.normal(0.0, 1.0) if 'delta' in param_names.keys() else 0.1,
        'mu_inf_SI': np.random.lognormal(0, var) if 'mu_inf_SI' in param_names.keys() else 1.0,
        'mu_inf_SC': np.random.lognormal(0, var) if 'mu_inf_SC' in param_names.keys() else 1.0,
        'mu_inf_AI': np.random.lognormal(0, var) if 'mu_inf_AI' in param_names.keys() else 1.0,
        'mu_inf_AC': np.random.lognormal(0, var) if 'mu_inf_AC' in param_names.keys() else 1.0,
        'mu_inf_AA': np.random.lognormal(0, var) if 'mu_inf_AA' in param_names.keys() else 1.0,
        'mu_susc_I': np.random.lognormal(0, var) if 'mu_susc_I' in param_names.keys() else 1.0,
        'mu_susc_C': np.random.lognormal(0, var) if 'mu_susc_C' in param_names.keys() else 1.0,
        'mu_protect_acq': np.random.lognormal(0, var) if 'mu_protect_acq' in param_names.keys() else 0.8,
        'mu_protect_transm': np.random.lognormal(0, var) if 'mu_protect_transm' in param_names.keys() else 1.0
    }
    return params


simulator_alpha = OutbreakSimulator(variant='alpha')
simulator_omicron = OutbreakSimulator(variant='omicron')

#%% md
# ## Generate training data
#%%
num_training_batches = 500
num_validation_sets = 5  # is multiplied by number of possible selection procedures, gives around 1000 datasets
training_data_file = f'models/training_data_{scenario_name}.pickle'
validation_data_file = f'models/valid_data_{scenario_name}.pickle'
#%%
@delayed
def simulate_single(generate_valid_data=False, selection_procedure_valid='all') -> dict:
    out = meta()
    out2 = prior(out['variant'])
    if out['variant'] == 'alpha':
        _simulator = simulator_alpha
    else:
        _simulator = simulator_omicron
    out.update(out2)

    if generate_valid_data:  # compute STAN posterior for validation data as well
        out3 = _simulator(selection_procedure=selection_procedure_valid, return_df=True, **out2)

        if selection_procedure_valid == 'all':
            for sp, val in out3.items():
                stan_posterior = get_stan_posterior(val['sim_data_df'], param_names, _simulator,
                                                    alpha=out['alpha'] if 'alpha' not in param_names else None,
                                                    #mu_protect_acq=out['mu_protect_acq'],
                                                    #mu_protect_transm=out['mu_protect_transm'],
                                                    show_progress=False)
                out3[sp].pop('sim_data_df')  # remove the DataFrame, we do not need it anymore
                for p_name in stan_posterior.keys():
                    out3[sp][f'stan_{p_name}'] = stan_posterior[p_name]
            for key in out3['random'].keys():
                out[key] = np.stack((out3['random'][key],
                                     out3['pedcov'][key],
                                     out3['adultcov'][key]), axis=0)
            # repeat entries
            out['variant'] = [[out['variant']], [out['variant']], [out['variant']]]
            out['variant_id'] = [[out['variant_id']], [out['variant_id']], [out['variant_id']]]
            out['selection_procedure'] = [['random'], ['pedcov'], ['adultcov']]
            out['selection_procedure_id'] = [[0], [1], [2]]  # 0 for random, 1 for pedcov, 2 for adultcov
            for p in full_name_list:
                out[p] = [[out[p]], [out[p]], [out[p]]]
            return out
        else:
            # just one selection procedure
            out['selection_procedure'] = selection_procedure_valid
            out['selection_procedure_id'] = 0 if selection_procedure_valid == 'random' else 1 if selection_procedure_valid == 'pedcov' else 2
            stan_posterior = get_stan_posterior(out3['sim_data_df'], param_names, _simulator,
                                                        alpha=out['alpha'] if 'alpha' not in param_names else None,
                                                        #mu_protect_acq=out['mu_protect_acq'],
                                                        #mu_protect_transm=out['mu_protect_transm'],
                                                        show_progress=False)
            out3.pop('sim_data_df')  # remove the DataFrame, we do not need it anymore
            for p_name in stan_posterior.keys():
                out3[f'stan_{p_name}'] = stan_posterior[p_name]
    else:
        out3 = _simulator(selection_procedure=out['selection_procedure'], **out2)

    out.update(out3)
    for p in full_name_list:
        out[p] = [out[p]]
    return out
# # #%%
# # test stan and simulations
# # stan uses 4 cpus per job, so we can use 1/4 of the available cpus
# test_data = Parallel(n_jobs=n_procs // 4, verbose=1)(simulate_single(generate_valid_data=True,
#                                                                 selection_procedure_valid='random') for _ in range(20))
# test_data = list_of_dicts_to_dict_of_lists(test_data)
# test_data_random = test_data.copy()
# select_id = test_data_random['selection_procedure'] == 'random'
# for k in test_data_random.keys():
#     test_data_random[k] = test_data_random[k][select_id.flatten()]
#
# stan_posterior_samples = {p: test_data_random[f'stan_{p}'] for p in param_names.keys()}
# fig = bf.diagnostics.recovery(stan_posterior_samples, test_data_random, variable_names=list(param_names.values()))
# fig.savefig('plots/test_recovery.png')
# fig = bf.diagnostics.calibration_ecdf(stan_posterior_samples, test_data_random, variable_names=list(param_names.values()), difference=True)
# fig.savefig('plots/test_ecdf.png')
# exit()

#%%
if os.path.exists(validation_data_file):
    # load simulation PedCov
    with open(validation_data_file, 'rb') as f:
        validation_data = pickle.load(f)
    try:
        with open(training_data_file, 'rb') as f:
            training_data = pickle.load(f)
    except FileNotFoundError:
        training_data = None
else:
    training_data = Parallel(n_jobs=n_procs, verbose=1)(simulate_single() for _ in range(batch_size * num_training_batches))
    training_data = list_of_dicts_to_dict_of_lists(training_data)
    with open(training_data_file, 'wb') as f:
        pickle.dump(training_data, f)

    # stan uses 4 cpus per job, so we can use 1/4 of the available cpus
    validation_data = Parallel(n_jobs=n_procs // 4, verbose=1)(simulate_single(generate_valid_data=True) for _ in range(batch_size * num_validation_sets))
    validation_data = list_of_dicts_to_dict_of_lists(validation_data)
    for key in validation_data.keys():  # concatenate the validation data for the selection procedures
        validation_data[key] = np.concatenate(validation_data[key], axis=0)
    with open(validation_data_file, 'wb') as f:
        pickle.dump(validation_data, f)
    exit()

# bring into same shape as training data
validation_data['variant'] = validation_data['variant'].flatten()
validation_data['variant_id'] = validation_data['variant_id'].flatten()
validation_data['selection_procedure'] = validation_data['selection_procedure'].flatten()
validation_data['selection_procedure_id'] = validation_data['selection_procedure_id'].flatten()


#%% md
# ## Neural Posterior Estimation
#%%
if len(variants) == 1:
    adapter = (
        bf.adapters.Adapter()
        .drop(['selection_procedure', 'variant'])  # drop strings
        .to_array()
        .one_hot('selection_procedure_id', num_classes=3)  # must be before convert_dtype
        .convert_dtype(from_dtype="float64", to_dtype="float32")

        .constrain('beta', lower=0, inclusive='none', method="softplus")  # standard
        .constrain([k for k in list(param_names.keys()) if k != 'delta' and k != 'beta'], lower=0, inclusive='none', method="exp")
        .concatenate(list(param_names.keys()), into="inference_variables")
        #.standardize('inference_variables')

        .rename('selection_procedure_id', to_key="inference_conditions")
        .rename('sim_data', to_key="summary_variables")

        .broadcast("inference_conditions", to="summary_variables", expand=(1,2))
        .concatenate(["summary_variables", "inference_conditions"], into="summary_variables")
    )
else:
    adapter = (
        bf.adapters.Adapter()
        .drop(['selection_procedure', 'variant'])  # drop strings
        .to_array()
        .one_hot('selection_procedure_id', num_classes=3)  # must be before convert_dtype
        .one_hot('variant_id', num_classes=2)  # must be before convert_dtype
        .convert_dtype(from_dtype="float64", to_dtype="float32")

        .constrain('beta', lower=0, inclusive='none', method="softplus")  # standard
        .constrain([k for k in list(param_names.keys()) if k != 'delta' and k != 'beta'], lower=0, inclusive='none', method="exp")
        .concatenate(list(param_names.keys()), into="inference_variables")
        #.standardize('inference_variables')

        .concatenate(['selection_procedure_id', 'variant_id'], into="inference_conditions")
        .rename('sim_data', to_key="summary_variables")

        # add inference conditions to summary variables
        .broadcast("inference_conditions", to="summary_variables", expand=(1,2))
        .concatenate(["summary_variables", "inference_conditions"], into="summary_variables")
    )

#%%
from bayesflow.utils.serialization import serializable

@serializable("bayesflow.networks")
class DoubleSummaryNetwork(bf.networks.SummaryNetwork):
    def __init__(self, inner_network, outer_network, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = 'inner_' + inner_network.name + '_outer_' + outer_network.name if name is None else name
        self.inner_network = inner_network  # operates over elements
        self.outer_network = outer_network  # operates over observations

    def call(self, x, training: bool = False, **kwargs):
        b_size, n_outer_obs, n_inner_obs = keras.ops.shape(x)[:3]

        # Flatten to combine batch and outer observation dimensions
        x_flat = keras.ops.reshape(x, (b_size * n_outer_obs, n_inner_obs, *keras.ops.shape(x)[3:]))

        # Apply the inner network to each element in the outer observation
        inner_output = self.inner_network(x_flat, training=training, **kwargs)

        # Reshape back to (b_size, n_outer_obs, inner_output_dim)
        inner_output = keras.ops.reshape(inner_output, (b_size, n_outer_obs, *keras.ops.shape(inner_output)[1:]))

        # Apply the outer network to the inner outputs
        outer_output = self.outer_network(inner_output, training=training, **kwargs)
        return outer_output


    def get_config(self):
        config = super().get_config()
        config.update({
            "inner_network": self.inner_network,
            "outer_network": self.outer_network
        })
        return config
#%%
model_id_list = itertools.product([0, 1, 2], [8, 16, 24, 32, 40], [16, 24, 32, 40])
sum_i = 0  # transformer summary
inf_i, in_summary_dim, out_summary_dim = list(model_id_list)[job_array_id]

if inf_i == 0:
    epochs = 100  # coupling flow
elif inf_i == 1:
    epochs = 300  # flow matching
elif inf_i == 2:
    epochs = 300  # consistency model
else:
    raise ValueError(f"Unknown inference network index: {inf_i}")

summary_network = [
    DoubleSummaryNetwork(
        inner_network=bf.networks.TimeSeriesNetwork(summary_dim=in_summary_dim, dropout=0.1, recurrent_dim=32),
        outer_network=bf.networks.SetTransformer(summary_dim=out_summary_dim, dropout=0.1),
        name=f'transformer_time_series'
    ),
    # DoubleSummaryNetwork(
    #     inner_network=bf.networks.TimeSeriesNetwork(summary_dim=in_summary_dim, dropout=0.1, recurrent_dim=32),
    #     outer_network=bf.networks.DeepSet(summary_dim=out_summary_dim, dropout=0.1,
    #                                       mlp_widths_equivariant=(128, 128)),
    #     name=f'deep_set_time_series'
    # )
][sum_i]

inference_network = [bf.networks.CouplingFlow(depth=7, subnet_kwargs={"dropout": 0.1}, transform='spline'),
                     bf.networks.FlowMatching(subnet_kwargs={"dropout": 0.1}, use_optimal_transport=True,
                                              integrate_kwargs={'method': 'rk45', 'steps': 'adaptive'}),
                     bf.networks.ConsistencyModel(epochs*num_training_batches),
                     ][inf_i]

model_name = (f'{scenario_name}_{["transformer", "deep_set"][sum_i]}_{["coupling_flow", "flow_matching", "consistency_model"][inf_i]}'
              f'_{in_summary_dim}_{out_summary_dim}.keras')

workflow = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=summary_network,
    inference_network=inference_network
)

model_path = f'models/{model_name}'
print(model_path)
history = workflow.fit_offline(
        data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
)
workflow.approximator.save(filepath=model_path)
diagnostics = workflow.plot_default_diagnostics(test_data=validation_data, num_samples=300,
                                                calibration_ecdf_kwargs={'difference': True})
diagnostics['losses'].savefig(f'plots/{scenario_name}_losses_{model_name}.pdf', bbox_inches='tight')
diagnostics['recovery'].savefig(f'plots/{scenario_name}_recovery_{model_name}.pdf', bbox_inches='tight')
diagnostics['calibration_ecdf'].savefig(f'plots/{scenario_name}_ecdf_{model_name}.pdf', bbox_inches='tight')

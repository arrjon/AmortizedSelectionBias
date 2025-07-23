#%% md
# # Simulation Based Inference to Remove Sampling Bias - Household Studies
# 
#%% md
#%%

# specify the PedCov path
data_path_alpha = 'PedCov/data/test_data_alpha.txt'  # todo: exchange with real PedCov
data_path_omicron = 'PedCov/data/test_data_omicron.txt'

# install dependencies
import install_requirements
#%%
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

job_array_id = 0  # int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_procs = 10  # int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
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
# We fix some parameters such that the model becomes identifiable (checked with STAN). We fix the parameters `alpha`.
#%%
import pickle
import itertools
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import keras
import bayesflow as bf

from PedCov.stan import get_stan_posterior
from PedCov.simulator import OutbreakSimulator
from PedCov.helper_functions import normalize_household_data, sampling_parameter_cis, sampling_parameter_cis_variants
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


#%%
simulator_alpha = OutbreakSimulator(variant='alpha')
simulator_omicron = OutbreakSimulator(variant='omicron')


#%%
colors = [  # colorblind safe colors
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#E15759",  # red
    "#76B7B2",  # teal
    "#59A14F",  # green
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray (neutral)
    "#8CD17D",  # light green
    "#FABFD2",  # light pink
]

stacked = True  # whether to stack the ECDFs or not
#%%

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
job_array_id = 14  # best model with normalizing flows is 14
model_id_list = itertools.product([0, 1, 2], [8, 16, 24, 32, 40], [16, 24, 32, 40])
nrmse_list = []
calibration_list = []
model_list = []
sum_i = 0  # transformer summary
for net_id, (inf_i, in_summary_dim, out_summary_dim) in enumerate(model_id_list):
    if net_id != job_array_id:
        continue
    if inf_i == 0:
        epochs = 100  # coupling flow
        #continue
    elif inf_i == 1:
        epochs = 300  # flow matching
        continue
    elif inf_i == 2:
        epochs = 300  # consistency model
        continue
    else:
        raise ValueError(f"Unknown inference network index: {inf_i}")

    summary_network = [
        DoubleSummaryNetwork(
             inner_network=bf.networks.TimeSeriesNetwork(summary_dim=in_summary_dim, dropout=0.1, recurrent_dim=32),
             outer_network=bf.networks.SetTransformer(summary_dim=out_summary_dim, dropout=0.1),
             name=f'transformer_time_series'
        ),
        DoubleSummaryNetwork(
             inner_network=bf.networks.TimeSeriesNetwork(summary_dim=in_summary_dim, dropout=0.1, recurrent_dim=32),
             outer_network=bf.networks.DeepSet(summary_dim=out_summary_dim, dropout=0.1,
                                               mlp_widths_equivariant=(128, 128)),
             name=f'deep_set_time_series'
        )
    ][sum_i]

    inference_network = [bf.networks.CouplingFlow(depth=7, subnet_kwargs={"dropout": 0.1}, transform='spline'),
                         bf.networks.FlowMatching(subnet_kwargs={"dropout": 0.1}, use_optimal_transport=True,
                                                  integrate_kwargs={'method': 'rk45', 'steps': 'adaptive'}),
                         #bf.networks.ConsistencyModel(epochs*num_training_batches),
                         ][inf_i]

    model_name = (f'{scenario_name}_{["transformer", "deep_set"][sum_i]}'
                  f'_{["coupling_flow", "flow_matching", "consistency_model"][inf_i]}'
                  f'_{in_summary_dim}_{out_summary_dim}.keras')

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network
    )

    model_path = f'models/{model_name}'
    print(model_path)
    if os.path.exists(model_path):
        workflow.approximator = keras.saving.load_model(filepath=model_path)
    else:
        raise NotImplementedError

#%% md
# # Apply trained model to real data
num_samples = 1000
#%%
# load the PedCov
dfs = {
    'alpha': pd.read_csv(data_path_alpha, delimiter=' ') if data_path_alpha is not None else None,
    'omicron': pd.read_csv(data_path_omicron, delimiter=' ') if data_path_omicron is not None else None
}

# patient id in household
dfs['alpha']['id_patient'] = dfs['alpha'].groupby("id_hh").cumcount() + 1  # Row number within id_hh
if data_path_omicron is not None:
    dfs['omicron']['id_patient'] = dfs['omicron'].groupby("id_hh").cumcount() + 1

real_data_results = {
    'alpha': None,
    'omicron': None
}

# prepare the PedCov for neural networks
for variant in variants:
    if variant == 'alpha':
        simulator_bf = simulator_alpha
    else:
        simulator_bf = simulator_omicron
    household_data = normalize_household_data(dfs[variant])[np.newaxis]
    real_data_results[variant] = {
        'sim_data': household_data,
        'selection_procedure': ['pedcov'],
        'selection_procedure_id': [1],  # pedcov
        'variant': [variant],
        'variant_id': [0] if variant == 'alpha' else [1],  # 0 for alpha, 1 for omicron
    }

# get posterior samples
for variant in variants:
    print(f"Variant: {variant}")
    if variant == 'alpha':
        simulator_bf = simulator_alpha
    else:
        simulator_bf = simulator_omicron

    posterior_samples_real = workflow.sample(conditions=real_data_results[variant], num_samples=num_samples)
    real_data_results[variant]['posterior_samples'] = posterior_samples_real

    stan_posterior_samples = get_stan_posterior(dfs[variant], param_names, simulator_bf, show_progress=True,
                                                alpha=0.001 if variant == 'alpha' else 0.01)

    # thin samples to num_samples
    sample_idx = np.random.choice(len(stan_posterior_samples[list(param_names.keys())[0]]), num_samples)
    for p in param_names.keys():
        stan_posterior_samples[p] = stan_posterior_samples[p][sample_idx].reshape(1, num_samples, 1)  # select samples
    real_data_results[variant]['stan_posterior_samples'] = stan_posterior_samples

# save samples
with open(f'plots/{scenario_name}_real_posterior_samples.pickle', 'wb') as f:
     pickle.dump(real_data_results, f)
#%%
# plot posterior samples vs prior
for variant in variants:
    print(f"Variant: {variant}")
    fig = bf.diagnostics.pairs_posterior(
        estimates=real_data_results[variant]['posterior_samples'],
        variable_keys=list(param_names.keys()),
        variable_names=list(param_names.values()),
        label_fontsize=22,
        legend_fontsize=24
    )
    plt.savefig(f'plots/{scenario_name}_real_posterior_{variant}.pdf', bbox_inches='tight')
    plt.show()

    fig = bf.diagnostics.pairs_posterior(
        estimates=real_data_results[variant]['stan_posterior_samples'],
        variable_keys=list(param_names.keys()),
        variable_names=list(param_names.values()),
        label_fontsize=22,
        legend_fontsize=24
    )
    plt.savefig(f'plots/{scenario_name}_real_stan_posterior_{variant}.pdf', bbox_inches='tight')
    plt.show()
#%%
# plot credible intervals for each parameter and each variant
for variant in variants:
    posterior_samples = np.stack([real_data_results[variant]['posterior_samples'][p][0, :, 0]
                                  for p in param_names.keys()], axis=-1)
    ax = sampling_parameter_cis(posterior_samples, alpha=[99, 95, 80], size=(7,3),
                                param_names=list(param_names.values()), title=f"Real Data NPE Posterior CIs - {variant}")
    # add vertical line at 1 for the mu parameters
    ax.vlines(1, ymin=1.75, ymax=8.25, color='grey', linestyle='--')
    ax.set_title("")
    ax.legend().remove()
    plt.savefig(f'plots/{scenario_name}_real_CIs_{variant}.pdf', bbox_inches='tight')
    plt.show()

    stan_posterior_samples = np.stack([real_data_results[variant]['stan_posterior_samples'][p][0, :, 0]
                                       for p in param_names.keys()], axis=-1)
    ax = sampling_parameter_cis(stan_posterior_samples, alpha=[99, 95, 80], size=(7,3),
                                param_names=list(param_names.values()), title=f"Real Data STAN Posterior CIs - {variant}")
    # add vertical line at 1 for the mu parameters
    ax.vlines(1, ymin=1.75, ymax=8.25, color='grey', linestyle='--')
    ax.set_title("")
    ax.legend().remove()
    plt.savefig(f'plots/{scenario_name}_real_stan_CIs_{variant}.pdf', bbox_inches='tight')
    plt.show()
#%%
ax = sampling_parameter_cis_variants(
    results={
        variant: real_data_results[variant] for variant in variants
    },
    variants=variants,
    param_dict=param_names,
    alpha=[99, 95, 80],
    size=(7, 3),
    key='posterior_samples',
    #show_legend=True
)
plt.savefig(f'plots/{scenario_name}_real_CIs_both_variants.pdf', bbox_inches='tight')
plt.show()

ax = sampling_parameter_cis_variants(
    results={
        variant: real_data_results[variant] for variant in variants
    },
    variants=variants,
    param_dict=param_names,
    alpha=[99, 95, 80],
    size=(7, 3),
    key='stan_posterior_samples',
)
plt.savefig(f'plots/{scenario_name}_real_stan_CIs_both_variants.pdf', bbox_inches='tight')
plt.show()
print("Done with real data analysis.")

# %%
# install dependencies
# import install_requirements

# %%
import os
os.environ['KERAS_BACKEND'] = 'jax'

import logging
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation as mad

import keras
import bayesflow as bf
from bayesflow.utils.serialization import serializable
from bayesflow.utils import filter_kwargs

from PedCov.stan import get_stan_posterior
from PedCov.helper_functions import (list_of_dicts_to_dict_of_lists, normalize_household_data,
                                     sampling_parameter_cis_comparison, plot_first_positive_age_group_counts)


try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path('/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/PedCov')
job_array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
n_procs = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))
partition = os.environ.get('SLURM_JOB_PARTITION', 'local')
batch_size = 64

method_colors = ['#4B2E83', '#1B8A8F']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

if 'gpu' in partition:
    OutbreakSimulator = lambda variant: None
else:
    from PedCov.simulator import OutbreakSimulator

variants = ['alpha', 'omicron']

# name of the scenario, used for saving files
if len(variants) == 2:
    scenario_name = 'pedcov'
elif variants[0] == 'alpha':
    scenario_name = f'pedcov_alpha'
elif variants[0] == 'omicron':
    scenario_name = f'pedcov_omicron'
else:
    raise ValueError(f"Unknown variant: {variants}")
logging.info(f'Scenario: {scenario_name}')

#%%
# ## Define model and prior
# 
# We fix some parameters such that the model becomes identifiable (checked with STAN). We fix the parameters `alpha`.

param_names = {  # comment out parameters that should not be estimated
    #'alpha': r'$\alpha$',
    'beta': r'$\beta$',
    'delta': r'$\delta$',
    'mu_inf_SI': r'$\mu_\text{inf}^\text{sym Infant}$',
    'mu_inf_SC': r'$\mu_\text{inf}^\text{sym Child}$',
    'mu_inf_AI': r'$\mu_\text{inf}^\text{asym Infant}$',
    'mu_inf_AC': r'$\mu_\text{inf}^\text{asym Child}$',
    'mu_inf_AA': r'$\mu_\text{inf}^\text{asym Adult}$',
    'mu_susc_I': r'$\mu_\text{sus}^\text{Infant}$',
    'mu_susc_C': r'$\mu_\text{sus}^\text{Child}$',
    'mu_protect_acq': r'$\mu_\text{pro}^\text{acq}$',
    'mu_protect_transm': r'$\mu_\text{pro}^\text{transm}$'
}

full_name_list = ['alpha', 'beta', 'delta',
                  'mu_inf_SI', 'mu_inf_SC', 'mu_inf_AI', 'mu_inf_AC', 'mu_inf_AA',
                  'mu_susc_I', 'mu_susc_C',
                  'mu_protect_acq', 'mu_protect_transm']

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

def prior(variant_name) -> dict:
    var = 0.7 # was 1 before
    if variant_name == 'alpha':
        alpha_fixed = 0.001  # fixed for alpha
    elif variant_name == 'omicron':
        alpha_fixed = 0.01  # fixed for omicron
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

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

#plot_incubation_distribution(simulator_alpha.shapeIncub, simulator_alpha.scaleIncub,
#                             simulator_alpha.shapeIncubAsymp, simulator_alpha.scaleIncubAsymp)
#plot_generation_time_distribution(simulator_alpha.shape_generation_time, simulator_alpha.scale_generation_time)
#plot_delay_distribution(simulator_alpha.delayDist)

# %%  Generate training data
num_training_batches = 500
num_validation_sets = 5  # is multiplied by number of possible selection procedures, gives around 1000 datasets
training_data_file = BASE / 'data' / f'training_data_{scenario_name}.pkl'
validation_data_file = BASE / 'data' / f'valid_data_{scenario_name}.pkl'

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
                                                        show_progress=False)
            out3.pop('sim_data_df')  # remove the DataFrame, we do not need it anymore
            for p_keys in stan_posterior.keys():
                out3[f'stan_{p_keys}'] = stan_posterior[p_keys]
    else:
        out3 = _simulator(selection_procedure=out['selection_procedure'], **out2)

    out.update(out3)
    for p in full_name_list:
        out[p] = [out[p]]
    return out

# # test stan and simulations
# # stan uses 4 cpus per job, so we can use 1/4 of the available cpus
# test_data = Parallel(n_jobs=10 // 4, verbose=1)(simulate_single(generate_valid_data=True,
#                                                                 selection_procedure_valid='random') for _ in range(2))
# test_data = list_of_dicts_to_dict_of_lists(test_data)
# test_data_random = test_data.copy()
# select_id = test_data_random['selection_procedure'] == 'random'
# for k in test_data_random.keys():
#     test_data_random[k] = test_data_random[k][select_id.flatten()]
#
# stan_posterior_samples = {p: test_data_random[f'stan_{p}'] for p in param_names.keys()}
# fig = bf.diagnostics.recovery(stan_posterior_samples, test_data_random, variable_names=list(param_names.values()))
# fig.savefig(BASE / 'plots' / 'test_recovery.png')
# fig = bf.diagnostics.calibration_ecdf(stan_posterior_samples, test_data_random, variable_names=list(param_names.values()), difference=True)
# fig.savefig(BASE / 'plots' / 'test_ecdf.png')

# %%
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
    training_data = Parallel(n_jobs=n_procs, verbose=1)(delayed(simulate_single)() for _ in range(batch_size * num_training_batches))
    training_data = list_of_dicts_to_dict_of_lists(training_data)
    with open(training_data_file, 'wb') as f:
        pickle.dump(training_data, f)

    # stan uses 4 cpus per job, so we can use 1/4 of the available cpus
    validation_data = Parallel(n_jobs=n_procs // 4, verbose=1)(delayed(simulate_single)(generate_valid_data=True) for _ in range(batch_size * num_validation_sets))
    validation_data = list_of_dicts_to_dict_of_lists(validation_data)
    for key in validation_data.keys():  # concatenate the validation data for the selection procedures
        validation_data[key] = np.concatenate(validation_data[key], axis=0)
    with open(validation_data_file, 'wb') as f:
        pickle.dump(validation_data, f)

# bring into same shape as training data
validation_data['variant'] = validation_data['variant'].flatten()
validation_data['variant_id'] = validation_data['variant_id'].flatten()
validation_data['selection_procedure'] = validation_data['selection_procedure'].flatten()
validation_data['selection_procedure_id'] = validation_data['selection_procedure_id'].flatten()

# get all random selection procedure data
validation_data_random = validation_data.copy()
select_id = validation_data_random['selection_procedure'] == 'random'
for k in validation_data_random.keys():
    validation_data_random[k] = validation_data_random[k][select_id.flatten()]

validation_data_pedcov = validation_data.copy()
select_id = validation_data_pedcov['selection_procedure'] == 'pedcov'
for k in validation_data_pedcov.keys():
    validation_data_pedcov[k] = validation_data_pedcov[k][select_id.flatten()]

validation_data_adultcov = validation_data.copy()
select_id = validation_data_adultcov['selection_procedure'] == 'adultcov'
for k in validation_data_adultcov.keys():
    validation_data_adultcov[k] = validation_data_adultcov[k][select_id.flatten()]

if training_data is not None:
    logging.info(f"Training data shape: {training_data['sim_data'].shape}")
logging.info(f"Validation data shape: {validation_data['sim_data'].shape}")
logging.info(f"Validation data Random: {validation_data_random['sim_data'].shape[0]}")
logging.info(f"Validation data PedCov: {validation_data_pedcov['sim_data'].shape[0]}")
logging.info(f"Validation data AdultCov: {validation_data_adultcov['sim_data'].shape[0]}")

# %%
parameter_colors = [  # colorblind safe colors
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

titles = {
    'random': 'Random selection procedure',
    'pedcov': 'Child selection procedure',
    'adultcov': 'Adult selection procedure'
}

if not 'gpu' in partition:
    for vd in [validation_data_random, validation_data_pedcov, validation_data_adultcov]:
        stan_posterior_samples = {p: vd[f'stan_{p}'] for p in param_names.keys()}

        # fig = bf.diagnostics.recovery(stan_posterior_samples, vd, variable_names=list(param_names.values()),
        #                               add_corr=False, figsize=(12, 5))
        # ax = fig.get_axes()
        # for i, a in enumerate(ax):
        #     if i != 7:
        #         a.set_xlabel("")
        #     if i > 1:
        #         a.set_xlim(0, 5)
        #         a.set_ylim(0, 5)
        # plt.tight_layout()
        # fig.savefig(BASE / 'plots' / f'{scenario_name}_recovery_stan_{vd["selection_procedure"][0]}.pdf', bbox_inches='tight')
        # plt.show()

        fig = bf.diagnostics.calibration_ecdf(stan_posterior_samples, vd, variable_names=list(param_names.values()),
                                              rank_ecdf_color=method_colors[0], label_fontsize=18, tick_fontsize=16,
                                              stacked=True, difference=True, figsize=(5, 3))
        ax = fig.get_axes()
        for i, a in enumerate(ax):
            a.get_legend().remove()
            lines = a.get_lines()
            handles = []
            for idx, line in enumerate(lines):
                if list(param_names.keys())[idx] in ['beta', 'delta']:
                    line.remove()
                    continue
                line.set_color(parameter_colors[idx])
                line.set_label(list(param_names.values())[idx])
                handles.append(Patch(facecolor=parameter_colors[idx], label=rf'{list(param_names.values())[idx]}'))
            handles += [Patch(facecolor='grey', label=r'$95\%$ Confidence Bands')]
            #a.legend(handles=handles, ncols=5, loc='lower center', bbox_to_anchor=(0.5, -0.9), frameon=False, fontsize=18)
            a.set_ylim(-0.4, 0.4)
            a.set_title(titles[vd['selection_procedure'][0]], fontsize=18)
            a.set_ylabel("MCMC ECDF difference", fontsize=18)
            #a.set_ylabel("")
            a.set_xlabel("")
        plt.tight_layout()
        fig.savefig(BASE / 'plots' / f'{scenario_name}_ecdf_stan_{vd["selection_procedure"][0]}.pdf', bbox_inches='tight')
        #fig.savefig(BASE / 'plots' / f'{scenario_name}_ecdf_legend.pdf', bbox_inches='tight')
        plt.show()
        #break

# %% Neural Posterior Estimation
if len(variants) == 1:
    adapter = (
        bf.adapters.Adapter()
        .drop(['selection_procedure', 'variant'])  # drop strings
        .to_array()
        .one_hot('selection_procedure_id', num_classes=3)  # must be before convert_dtype
        .convert_dtype(from_dtype="float64", to_dtype="float32")

        .log([k for k in list(param_names.keys()) if k != 'delta'])
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

        .log([k for k in list(param_names.keys()) if k != 'delta'])
        .concatenate(list(param_names.keys()), into="inference_variables")

        .concatenate(['selection_procedure_id', 'variant_id'], into="inference_conditions")

        .rename('sim_data', to_key="summary_variables")

        # add inference conditions to summary variables
        .broadcast("inference_conditions", to="summary_variables", expand=(1,2))
        .concatenate(["summary_variables", "inference_conditions"], into="summary_variables")
    )

#%%
test_data = adapter.forward(validation_data)['inference_variables']
#test_data = adapter.forward(validation_data)['summary_variables']

fig, ax = plt.subplots(3, test_data.shape[-1] // 2, figsize=(8, 3), layout='constrained')
ax = ax.flatten()
for i in range(test_data.shape[-1]):
    ax[i].hist(test_data[:, i].flatten(), bins=30, density=True)
    #ax[i].plot(test_data[:, i])
    ax[i].set_title(f'Condition {i}')
plt.show()
print(test_data.shape, np.isnan(test_data).any())

#%%
@serializable("bayesflow.networks")
class DoubleSummaryNetwork(bf.networks.SummaryNetwork):
    def __init__(self, inner_network, outer_network, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = 'inner_' + inner_network.name + '_outer_' + outer_network.name if name is None else name
        self.inner_network = inner_network  # operates over elements
        self.outer_network = outer_network  # operates over observations
        self.use_mask = False
        if 'transformer' in inner_network.name:
            self.use_mask = True

    def call(self, x, training: bool = False, **kwargs):
        b_size, n_outer_obs, n_inner_obs = keras.ops.shape(x)[:3]

        # Flatten to combine batch and outer observation dimensions
        x_flat = keras.ops.reshape(x, (b_size * n_outer_obs, n_inner_obs, *keras.ops.shape(x)[3:]))

        # Apply the inner network to each element in the outer observation
        if self.use_mask:
            member_mask = self._compute_member_mask(x_flat)
            inner_output = self.inner_network(x_flat, training=training,
                                              **filter_kwargs(kwargs | {"attention_mask": member_mask}, self.inner_network.call))
        else:
            inner_output = self.inner_network(x_flat, training=training, **filter_kwargs(kwargs, self.inner_network.call))

        # Reshape back to (b_size, n_outer_obs, inner_output_dim)
        inner_output = keras.ops.reshape(inner_output, (b_size, n_outer_obs, *keras.ops.shape(inner_output)[1:]))

        # Apply the outer network to the inner outputs
        outer_output = self.outer_network(inner_output, training=training, **filter_kwargs(kwargs, self.outer_network.call))

        return outer_output


    def get_config(self):
        config = super().get_config()
        config.update({
            "inner_network": self.inner_network,
            "outer_network": self.outer_network,
        })
        return config

    @staticmethod
    def _compute_member_mask(x, ignore_features_from_idx=11):
        """Compute mask for valid household members (non-padding)"""
        # Check if all features are zero for each member
        feature_sum = keras.ops.sum(keras.ops.abs(x[..., :ignore_features_from_idx]), axis=-1)

        # Mask is True where sum > epsilon for numerical stability
        epsilon = 1e-8
        mask_1d = feature_sum > epsilon

        mask = keras.ops.logical_and(
            keras.ops.expand_dims(mask_1d, axis=1),  # (B, 1, T)
            keras.ops.expand_dims(mask_1d, axis=2)  # (B, S, 1)
        )
        return mask  # (B, T, S)

# %%
if job_array_id == -1:
    EPOCHS = 2
    summary_network = DoubleSummaryNetwork(
        inner_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        outer_network=bf.networks.DeepSet(summary_dim=len(param_names) * 3, dropout=0.1),
    )
    inference_network = bf.networks.CouplingFlow(depth=2)
    network_name = 'test'
    training_data = {}
    for k, v in validation_data.items():
        training_data[k] = v[:100]
elif job_array_id == 0:
    EPOCHS = 100
    summary_network = DoubleSummaryNetwork(
        inner_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        outer_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        name=f'transformer_transformer'
    )
    inference_network = bf.networks.CouplingFlow(depth=7, subnet_kwargs={"dropout": 0.1}, transform='spline')
    network_name = 'coupling_flow_tt'
elif job_array_id == 1:
    EPOCHS = 150
    summary_network = DoubleSummaryNetwork(
        inner_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        outer_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        name=f'transformer_transformer'
    )
    inference_network = bf.networks.FlowMatching(subnet_kwargs={"dropout": 0.1})
    network_name = 'flow_matching_tt_150'
elif job_array_id == 2:
    EPOCHS = 300
    summary_network = DoubleSummaryNetwork(
        inner_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        outer_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        name=f'transformer_transformer'
    )
    inference_network = bf.networks.FlowMatching(subnet_kwargs={"dropout": 0.1})
    network_name = 'flow_matching_tt_300'
elif job_array_id == 3:
    EPOCHS = 500
    summary_network = DoubleSummaryNetwork(
        inner_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        outer_network=bf.networks.SetTransformer(summary_dim=len(param_names) * 3, dropout=0.1),
        name=f'transformer_transformer'
    )
    inference_network = bf.networks.FlowMatching(subnet_kwargs={"dropout": 0.1})
    network_name = 'flow_matching_tt_500'
else:
    raise ValueError(f"Unknown job array id: {job_array_id}")

model_path = BASE / 'models' / f'{scenario_name}_{network_name}.keras'
logging.info(model_path)

workflow = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=summary_network,
    inference_network=inference_network
)

#%%#
class HistoryClass(object):
    def __init__(self, history_to_save):
        self.history = history_to_save
if os.path.exists(model_path):
    workflow.approximator = keras.saving.load_model(filepath=model_path)

    try:
        with open(BASE / 'models' / f'history_{network_name}.pkl', 'rb') as file:
            workflow.history = pickle.load(file)
    except FileNotFoundError:
        logging.info("No history file found.")
else:
    history = workflow.fit_offline(
        data=training_data,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_data=validation_data,
    )
    workflow.approximator.save(filepath=model_path)

    diagnostics = workflow.compute_default_diagnostics(
        test_data=validation_data, num_samples=500, approximator_kwargs=dict(batch_size=batch_size)
    )
    logging.info(f"RMSE {diagnostics.loc['NRMSE'].mean()}")
    logging.info(f"Calibration Error {diagnostics.loc['Calibration Error'].mean()}")

    with open(BASE / 'models' / f'history_{network_name}.pkl', 'wb') as file:
        model_history = HistoryClass(history.history)
        pickle.dump(model_history, file)

#%%
diagnostics = workflow.plot_default_diagnostics(
    test_data=validation_data, num_samples=500, approximator_kwargs=dict(batch_size=batch_size),
    loss_kwargs=dict(val_color=method_colors[-1], train_color='black'),
    recovery_kwargs=dict(color=method_colors[-1]),
    calibration_ecdf_kwargs=dict(rank_ecdf_color=method_colors[-1]),
    coverage_kwargs=dict(color=method_colors[-1]),
    z_score_contraction_kwargs=dict(color=method_colors[-1]),
    variable_names=list(param_names.values())
)
for diagnostic, fig_d in diagnostics.items():
    fig_d.savefig(BASE / 'plots' / f'{scenario_name}_{network_name}_{diagnostic}.pdf', bbox_inches='tight')
    plt.close(fig_d)

#%%
logging.info('Prepare C2ST classifier')
embedded_data = []
for i in range(0, len(validation_data['sim_data']), batch_size):
    batch = {name: validation_data[name][i:i + batch_size] for name in validation_data}
    embedded_data_batch = workflow.approximator.summarize(batch)
    embedded_data.append(embedded_data_batch)
embedded_data = np.vstack(embedded_data)
target_params = np.concatenate([validation_data[k] for k in param_names], axis=-1)
targets = np.concatenate((target_params, embedded_data), axis=-1)
posterior_samples_test = workflow.sample(conditions=validation_data, num_samples=1, batch_size=batch_size)
posterior_samples_test = np.concatenate([posterior_samples_test[k][:, 0]  for k in param_names], axis=-1)
estimates = np.concatenate((posterior_samples_test, embedded_data), axis=-1)
estimates_mean = np.mean(estimates, axis=0)
estimates_std = np.std(estimates, axis=0)
estimates = (estimates - estimates_mean) / estimates_std
targets = (targets - estimates_mean) / estimates_std
logging.info("Train C2ST classifier")
c2st_results = bf.diagnostics.metrics.classifier_two_sample_test(
    estimates=estimates,
    targets=targets,
    return_metric_only=False,
    batch_size=batch_size,
    standardize=False
)
logging.info(f'C2ST Accuracy: {c2st_results["score"]}')

logging.info('Train C2ST random classifiers')
c2st_results_random = []
full_set = np.concatenate((estimates, targets), axis=0)
for _ in range(10):
    # permute all labels to create random classifier
    np.random.shuffle(full_set)  # shuffles the array along the first axis of a multi-dimensional array
    estimates_random = full_set[:estimates.shape[0]]
    targets_random = full_set[estimates.shape[0]:]

    c2st_results_random.append(bf.diagnostics.metrics.classifier_two_sample_test(
        estimates=estimates_random,
        targets=targets_random,
        return_metric_only=False,
        batch_size=batch_size,
        cross_validation_splits=0,
        validation_split=0.1,
        standardize=False
    ))

# %% Simulation Study on Validation Data
for vd in [validation_data_random, validation_data_pedcov, validation_data_adultcov]:
    posterior_samples = workflow.sample(num_samples=1000, conditions=vd, batch_size=batch_size)

    # fig = bf.diagnostics.recovery(posterior_samples, vd, variable_names=list(param_names.values()),
    #                               add_corr=False, figsize=(12, 5))
    # ax = fig.get_axes()
    # for i, a in enumerate(ax):
    #     if i != 7:
    #         a.set_xlabel("")
    #     if i > 1:
    #         a.set_xlim(0, 5)
    #         a.set_ylim(0, 5)
    # plt.tight_layout()
    # fig.savefig(BASE / 'plots' / f'{scenario_name}_recovery_npe_{vd["selection_procedure"][0]}.pdf', bbox_inches='tight')
    # plt.show()

    fig = bf.diagnostics.calibration_ecdf(posterior_samples, vd, variable_names=list(param_names.values()),
                                          rank_ecdf_color=method_colors[-1], label_fontsize=18, tick_fontsize=16,
                                          stacked=True, difference=True, figsize=(5, 3))
    ax = fig.get_axes()
    for i, a in enumerate(ax):
        a.get_legend().remove()
        lines = a.get_lines()
        for idx, line in enumerate(lines):
            if list(param_names.keys())[idx] in ['beta', 'delta']:
                line.remove()
            line.set_color(parameter_colors[idx])
            line.set_label(f'{list(param_names.values())[idx]}')
        a.set_ylim(-0.4, 0.4)
        a.set_title(titles[vd['selection_procedure'][0]], fontsize=18)
        a.set_ylabel("NPE ECDF difference", fontsize=18)
    plt.tight_layout()
    fig.savefig(BASE / 'plots' / f'{scenario_name}_{network_name}_ecdf_npe_{vd["selection_procedure"][0]}.pdf',
                bbox_inches='tight')
    plt.show()

#%%
# create test data to validate code
# test_params = prior("alpha")
# test_df = simulator_alpha(**test_params, return_df=True)['sim_data_df']
# test_df.to_csv(BASE / 'data' / 'test_data_alpha.txt', sep=' ', index=False)
#
# test_params = prior("omicron")
# test_df = simulator_omicron(**test_params, return_df=True)['sim_data_df']
# test_df.to_csv(BASE / 'data' / 'test_data_omicron.txt', sep=' ', index=False)

# specify the PedCov path
data_path_alpha = BASE / 'data' / 'real_data_alpha.txt'
data_path_omicron = BASE / 'data' / 'real_data_omicron.txt'
num_samples = 1000

if os.path.exists(BASE / 'data' / f'alpha_sim_data.npy'):
    logging.info("Loading prepared real data and stan posterior samples from file")
    real_data_results = {}
    for variant in variants:
        household_data = np.load(BASE / 'data' / f'{variant}_sim_data.npy')
        with open(BASE / 'stan' / f'{variant}_stan_posterior_samples.pkl', 'rb') as f:
            stan_posterior_samples = pickle.load(f)

        real_data_results[variant] = {
            'sim_data': household_data,
            'selection_procedure': ['pedcov'],
            'selection_procedure_id': [1],  # pedcov
            'variant': [variant],
            'variant_id': [0] if variant == 'alpha' else [1],  # 0 for alpha, 1 for omicron
            'stan_posterior_samples': stan_posterior_samples
        }
else:
    # prepare real data
    dfs = {
        'alpha': pd.read_csv(data_path_alpha, delimiter=' ') if data_path_alpha is not None else None,
        'omicron': pd.read_csv(data_path_omicron, delimiter=' ') if data_path_omicron is not None else None
    }

    # patient id in household
    dfs['alpha']['id_patient'] = dfs['alpha'].groupby("id_hh").cumcount() + 1  # Row number within id_hh
    if data_path_omicron is not None:
        dfs['omicron']['id_patient'] = dfs['omicron'].groupby("id_hh").cumcount() + 1

    real_data_results = {
        'alpha': {},
        'omicron': {}
    }

    # prepare the PedCov for neural networks
    for variant in variants:
        if variant == 'alpha':
            simulator_bf = simulator_alpha
        else:
            simulator_bf = simulator_omicron
        household_data = normalize_household_data(dfs[variant])[np.newaxis]
        np.save(BASE / 'data' / f'{variant}_sim_data.npy', household_data)

        logging.info("Computing STAN posterior samples for real data")
        stan_posterior_samples = get_stan_posterior(dfs[variant], param_names, simulator_bf, show_progress=True,
                                                    alpha=0.001 if variant == 'alpha' else 0.01)
        # thin samples to num_samples
        sample_idx = np.random.choice(len(stan_posterior_samples[list(param_names.keys())[0]]), num_samples)
        for p in param_names.keys():
            stan_posterior_samples[p] = stan_posterior_samples[p][sample_idx].reshape(1, num_samples, 1)  # select samples

        with open(BASE / 'stan' / f'{variant}_stan_posterior_samples.pkl', 'wb') as f:
            pickle.dump(stan_posterior_samples, f)

        real_data_results[variant] = {
            'sim_data': household_data,
            'selection_procedure': ['pedcov'],
            'selection_procedure_id': [1],  # pedcov
            'variant': [variant],
            'variant_id': [0] if variant == 'alpha' else [1],  # 0 for alpha, 1 for omicron
            'stan_posterior_samples': stan_posterior_samples
        }

plot_first_positive_age_group_counts(
    [validation_data_random, validation_data_pedcov, validation_data_adultcov],
    colors=method_colors+['#D64A62'],
    labels=['Random Selection', 'Child selection', 'Adult selection'],
    real_data=real_data_results,
    save_path=BASE / "plots" / "first_positive_age_group_counts.pdf",
)

#%% NPE inference on real data
c2st_result_real_random = []
for variant in variants:
    posterior_file = BASE / 'models' / f'{variant}_{network_name}_npe_posterior_samples.pkl'
    if os.path.exists(posterior_file):
        # load results
        with open(posterior_file, 'rb') as f:
            real_data_results = pickle.load(f)
        logging.info("Loaded posterior samples from file")
    else:
        logging.info(f"Computing posterior samples for real data {variant}")

        prep_dict = real_data_results[variant].copy()
        prep_dict.pop('stan_posterior_samples')  # remove stan samples for conditions, adapter cannot handle them
        if 'posterior_samples' in prep_dict.keys():
            prep_dict.pop('posterior_samples')
        posterior_samples_real = workflow.sample(
            conditions=prep_dict, num_samples=num_samples, batch_size=batch_size
        )
        real_data_results[variant]['posterior_samples'] = posterior_samples_real

        embedded_real_data = workflow.approximator.summarize(prep_dict)
        embedded_real_data = np.repeat(embedded_real_data, repeats=num_samples, axis=0)
        posterior_samples_test = np.concatenate([posterior_samples_real[k][0] for k in param_names], axis=-1)
        estimates_real = np.concatenate((posterior_samples_test, embedded_real_data), axis=-1)
        estimates_real = (estimates_real - estimates_mean) / estimates_std
        scores = np.array([c.predict(estimates_real).flatten() for c in c2st_results['classifiers']])
        scores = np.maximum(scores, 1 - scores)
        c2st_score = np.mean(scores, axis=0)
        test_statistic = np.mean((c2st_score - 0.5) ** 2)
        real_data_results[variant]['C2ST'] = c2st_score
        logging.info(f'Real Data {variant} C2ST Accuracy: {np.mean(real_data_results[variant]["C2ST"])}')

        # apply random classifiers
        scores_random = np.array([c['classifiers'][0].predict(estimates_real).flatten() for c in c2st_results_random])
        scores_random = np.maximum(scores_random, 1 - scores_random)
        test_statistic_random = np.mean((scores_random - 0.5) ** 2, axis=-1)
        p_val = np.mean(test_statistic_random > test_statistic)
        c2st_result_real_random.append((test_statistic, p_val))
        logging.info(f'C2ST Statistic: {test_statistic}, p-value: {p_val}')

        del prep_dict
        # save samples
        with open(posterior_file, 'wb') as f:
            pickle.dump(real_data_results, f)
        logging.info(f'Variant {variant} bias-aware NPE Posterior Samples:')
        for k, v in posterior_samples_real.items():
            logging.info(f'{k}: median={np.median(v):.3f}, mad={mad(v.flatten()):.3f}')
        logging.info(f'Variant {variant} MCMC Posterior Samples:')
        for k, v in real_data_results[variant]['stan_posterior_samples'].items():
            logging.info(f'{k}: median={np.median(v):.3f}, mad={mad(v.flatten()):.3f}')

#%%
if len(variants) == 2:
    fig, axis = plt.subplots(ncols=len(variants), sharex=True, sharey=True,
                             figsize=(7, 2.9), layout='constrained')
    axis[0], _ = sampling_parameter_cis_comparison(
            results=real_data_results['alpha'],
            methods={'posterior_samples': 'Bias-aware NPE', 'stan_posterior_samples': 'MCMC'},
            variant='alpha',
            param_dict=param_names,
            alpha=[99, 95, 80],
            size=(5, 4),
            show_legend=False,
            colors=method_colors[::-1],
            ax=axis[0]
        )
    axis[1], handles = sampling_parameter_cis_comparison(
            results=real_data_results['omicron'],
            methods={'posterior_samples': 'Bias-aware NPE', 'stan_posterior_samples': 'MCMC'},
            variant='omicron',
            param_dict=param_names,
            alpha=[99, 95, 80],
            size=(5, 4),
            show_legend=False,
            colors=method_colors[::-1],
        ax=axis[1]
        )
    axis[1].set_ylabel("")
    for ax in axis:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    handles = [handles[-2], handles[-1]]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(BASE / 'plots' / f'{scenario_name}_{network_name}_real_CIs.pdf', bbox_inches='tight')
    plt.show()

#%% apply C2ST
if len(c2st_result_real_random) > 0:
    logging.info("Plotting C2ST histograms for real data")
    bins = 20
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "Reds_trunc",
        plt.cm.Reds(np.linspace(0.1, 1.0, 256))
    )
    fig, ax = plt.subplots(nrows=len(param_names), ncols=len(variants), sharey='row', sharex='row', figsize=(10, 12),
                           layout='constrained')
    for p_i, (p_name, p_name_pretty) in enumerate(param_names.items()):
        for v_i, variant in enumerate(variants):
            # compute bin assignment
            x = real_data_results[variant]['posterior_samples'][p_name].flatten()
            counts, bin_edges = np.histogram(x, bins=bins, density=True)
            bin_idx = np.digitize(x, bin_edges) - 1

            # compute mean color per bin
            bin_color = np.array([
                np.mean(real_data_results[variant]['C2ST'][bin_idx == i]) if np.any(bin_idx == i) else 0
                for i in range(bins)
            ])

            # plot histogram manually
            for i in range(bins):
                ax[p_i, v_i].bar(
                    bin_edges[i],
                    counts[i],
                    width=bin_edges[i + 1] - bin_edges[i],
                    align="edge",
                    color=cmap(norm(bin_color[i])),
                    edgecolor=cmap(norm(bin_color[i]))
                )

            ax[p_i, v_i].set_xlabel(p_name_pretty)
            if v_i == 0:
                ax[p_i, v_i].set_ylabel("Density")
            if p_i == 0:
                ax[p_i, v_i].set_title(rf"Variant {variant}")
                m_score = np.mean(real_data_results[variant]['C2ST'])
                ax[p_i, v_i].text(
                    0.95, 0.95,
                    f"Mean C2ST={m_score:.2f}\np-value={c2st_result_real_random[v_i][1]:.2f}",
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax[p_i, v_i].transAxes,
                    fontsize=9,
                )
            # remove top and right spines
            ax[p_i, v_i].spines['top'].set_visible(False)
            ax[p_i, v_i].spines['right'].set_visible(False)

    # add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), label="C2ST score (mean per bin)", fraction=0.02)
    fig.savefig(BASE / 'plots' / f'{scenario_name}_{network_name}_real_c2st_histograms.pdf', bbox_inches='tight')
    plt.show()

logging.info("Done!")

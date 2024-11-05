import logging
import os
import pickle
from functools import partial
from itertools import product
from time import perf_counter
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from bayesflow import default_settings as defaults
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.helper_networks import MultiConv1D
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from tensorflow.keras.layers import Dense, GRU, Bidirectional
from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CustomLoader:
    def __init__(self, file_path: str, iterations_per_epoch: int):
        self.file_path = file_path
        self.iterations_per_epoch = iterations_per_epoch
        self.files = os.listdir(self.file_path)
        np.random.shuffle(self.files)  # Shuffle file order initially
        self.remaining_files = list(self.files)

    def get_next_iteration(self, iterations_per_epoch: int = None) -> list:
        """Loads files dynamically to return a batch of simulations"""
        if iterations_per_epoch is None:
            iterations_per_epoch = self.iterations_per_epoch
        iteration_batches = []

        # Load until we have enough for the batch or no more files are left
        while len(iteration_batches) < iterations_per_epoch and self.remaining_files:
            # Pop one file from the remaining files
            file = self.remaining_files.pop()
            with open(os.path.join(self.file_path, file), 'rb') as batch_file:
                simulation_batch = pickle.load(batch_file)[0]
                assert isinstance(simulation_batch, dict)   # only one batch per file
                iteration_batches.append(simulation_batch)

        # If the batch is still smaller than the required batch_size, reshuffle and continue
        if len(iteration_batches) < iterations_per_epoch:
            self.remaining_files = list(self.files)
            np.random.shuffle(self.remaining_files)
            # Recursively call to complete the iteration
            iteration_batches.extend(self.get_next_iteration(iterations_per_epoch - len(iteration_batches)))

        return iteration_batches

    def __call__(self, file_path=None) -> list:
        # file_path not needed, just to match the signature of the loader
        return self.get_next_iteration(self.iterations_per_epoch)


alpha_community_infection = np.array([0.0005, 0.002, 0.003])
omicron_community_infection = alpha_community_infection * 10
community = np.array([alpha_community_infection, omicron_community_infection])


def configurator(forward_dict: dict,
                 prior_mean: np.ndarray, prior_std: np.ndarray,
                 not_inform_selection: str = None, keep_selection: str = None, drop_n_households = False) -> dict:
    out_dict = {}

    # Extract data (already normalized)
    x = forward_dict["sim_data"]
    if drop_n_households:
        if isinstance(drop_n_households, bool):
            drop_n_households = np.random.choice([0., 0.1, 0.3, 0.5, 0.7, 0.9], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        drop_n_households = int(drop_n_households * x.shape[1])
        logger.info(f'Dropping {drop_n_households} / {x.shape[1]} households')
        x = x[:, drop_n_households:]  # order of households is random anyway
    out_dict['summary_conditions'] = x.astype(np.float32)

    # Extract params
    if 'parameters' in forward_dict.keys():
        forward_dict["prior_draws"] = forward_dict["parameters"]
    if 'prior_draws' in forward_dict.keys():
        params = forward_dict["prior_draws"]
        # normalize
        params = (params - prior_mean) / prior_std
        out_dict['parameters'] = params.astype(np.float32)

    # Extract context
    # non batchable context can come as a list or as a single value
    if isinstance(forward_dict['sim_non_batchable_context'], str):
        forward_dict['sim_non_batchable_context'] = [forward_dict['sim_non_batchable_context']] * len(x)
    sim_non_batchable_context = np.array(forward_dict['sim_non_batchable_context']).flatten()  # one value per sample
    if len(forward_dict['sim_batchable_context']) == 2:
        if not isinstance(forward_dict['sim_batchable_context'][0], str):
            raise ValueError('"sim_batchable_context" has an unexpected format')
        forward_dict['sim_batchable_context'] = [forward_dict['sim_batchable_context']] * len(x)
    sim_batchable_context = np.array(forward_dict['sim_batchable_context'])  # two values per sample

    direct_condition = np.zeros((len(x), 3), dtype=np.float32)
    direct_condition[sim_non_batchable_context == 'omicron', 0] = 1.  #  if variant is alpha=0, omicron=1
    direct_condition[sim_batchable_context[:, 0] == 'random', 1] = 1.  #  if selection procedure is pedcov=0, random=1
    direct_condition[:, 2] = (np.log(sim_batchable_context[:, 1].astype(np.float32)) - community.mean()) / community.std()  # alpha value

    if keep_selection is not None:
        logger.warning(f'Drop all but {keep_selection}')
        not_inform_selection = keep_selection
        # Extract context
        keep_indices = np.array(sim_batchable_context[:, 0] == keep_selection)
        if keep_indices.size == 0:
            # If keep_indices is empty, select a random index to not have an empty batch
            logger.warning("keep_indices is empty, select a random index")
            keep_indices = np.array([np.random.choice(len(sim_batchable_context))])

        direct_condition = direct_condition[keep_indices]
        out_dict['summary_conditions'] = out_dict['summary_conditions'][keep_indices]
        if 'parameters' in out_dict.keys():
            out_dict['parameters'] = out_dict['parameters'][keep_indices]

    if not_inform_selection is not None:
        # drop second direct condition on selection procedure
        if keep_selection is None:
            # no warning issued before
            logger.info('Warning: Not informing about selection procedure')
        direct_condition = direct_condition[:, [0, 2]]

    out_dict['direct_condition'] = direct_condition.astype(np.float32)

    assert out_dict['summary_conditions'].shape[0] == out_dict['direct_condition'].shape[0], \
        'Number of samples in summary conditions and direct condition do not match'
    if 'parameters' in out_dict.keys():
        assert out_dict['summary_conditions'].shape[0] == out_dict['parameters'].shape[0], \
            'Number of samples in summary conditions and parameters do not match'
    return out_dict


# configurator for joint inference, no conditioning on variant, relative priors
def configurator_joint(forward_dict: dict,
                       prior_mean: np.ndarray, prior_std: np.ndarray,
                       make_prior_relative: callable,
                       not_inform_selection: str = None, keep_selection: str = None) -> dict:
    out_dict = {}

    # Extract data (already normalized)
    x = forward_dict["sim_data"]
    out_dict['summary_conditions'] = x.astype(np.float32)

    # Extract params
    if 'parameters' in forward_dict.keys():
        forward_dict["prior_draws"] = forward_dict["parameters"]
    if 'prior_draws' in forward_dict.keys():
        params = forward_dict["prior_draws"]
        params = make_prior_relative(params)
        # normalize
        params = (params - prior_mean) / prior_std
        out_dict['parameters'] = params.astype(np.float32)


    # Extract context
    if len(forward_dict['sim_batchable_context']) == 2:
        if not isinstance(forward_dict['sim_batchable_context'][0], str):
            raise ValueError('"sim_batchable_context" has an unexpected format')
        forward_dict['sim_batchable_context'] = [forward_dict['sim_batchable_context']] * len(x)

    selection_procedure = np.array([sel for sel, _ in forward_dict['sim_batchable_context']])  # two values per sample
    alpha_condition = np.array([a for _, a in forward_dict['sim_batchable_context']]).astype(np.float32)
    direct_condition = np.zeros((len(x), 3), dtype=np.float32)
    direct_condition[selection_procedure == 'random', 0] = 1.  # if selection procedure is pedcov=0, random=1
    direct_condition[:, 1:] = (np.log(alpha_condition) - community.mean()) / community.std()  # alpha value

    if keep_selection is not None:
        logger.warning(f'Drop all but {keep_selection}')
        not_inform_selection = keep_selection
        # Extract context
        keep_indices = np.array(selection_procedure == keep_selection)
        if keep_indices.size == 0:
            # If keep_indices is empty, select a random index to not have an empty batch
            logger.warning("keep_indices is empty, select a random index")
            keep_indices = np.array([np.random.choice(len(selection_procedure))])

        direct_condition = direct_condition[keep_indices]
        out_dict['summary_conditions'] = out_dict['summary_conditions'][keep_indices]
        if 'parameters' in out_dict.keys():
            out_dict['parameters'] = out_dict['parameters'][keep_indices]

    if not_inform_selection is not None:
        # drop direct condition on selection procedure
        if keep_selection is None:
            # no warning issued before
            logger.info('Warning: Not informing about selection procedure')
        out_dict['direct_condition'] = direct_condition[:, 1][:, np.newaxis].astype(np.float32)
    else:
        out_dict['direct_condition'] = direct_condition.astype(np.float32)

    if 'parameters' in out_dict.keys():
        assert out_dict['summary_conditions'].shape[0] == out_dict['parameters'].shape[0], \
            'Number of samples in summary conditions and parameters do not match'
    return out_dict


# define the network
class GroupSummaryNetwork(tf.keras.Model):
    """Network to summarize the data of groups of cells.  Each group is passed through a series of convolutional layers
    followed by an LSTM layer. The output of the LSTM layer is then pooled across the groups and dense layer applied
    to obtain a summary of fixed dimensionality. The network is invariant to the order of the groups.
    """

    def __init__(
            self,
            summary_dim,
            use_time_attention,
            num_conv_layers=2,
            rnn_units=32,
            bidirectional=True,
            conv_settings=None,
            return_attention_weights=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.use_time_attention = use_time_attention
        self.return_attention_weights = return_attention_weights

        if self.use_time_attention:
            # no masking, full time series used, then attention on time series
            if conv_settings is None:
                conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV
            conv = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])
            self.group_conv = tf.keras.layers.TimeDistributed(conv)
            self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=rnn_units)
        self.norm_layer = tf.keras.layers.LayerNormalization()

        if bidirectional:
            rnn = Bidirectional(GRU(rnn_units, return_sequences=use_time_attention))
        else:
            rnn = GRU(rnn_units, return_sequences=use_time_attention)
        self.group_rnn = tf.keras.layers.TimeDistributed(rnn)

        self.out_layer = Dense(units=summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def call(self, x, **kwargs):
        """Performs a forward pass through the network by first passing `x` through the same rnn network for
        each household and then pooling the outputs across households.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_groups, n_time_steps, n_features)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, summary_dim)
        """
        attention_weights = None

        # Apply the RNN to each group
        if self.use_time_attention:
            # no masking, full time series used, then attention on time series
            out = self.group_conv(x, **kwargs)
            out = self.group_rnn(out, **kwargs)
            #  return full sequence (batch_size, n_groups, n_time_steps, lstm_units)
            # bidirectional LSTM returns 2*lstm_units

            # learn a query vector to attend over the time points (mean over groups)
            query = tf.reduce_mean(out, axis=1, keepdims=True)  # Shape: (batch_size, 1, n_time_steps, lstm_units)

            if not self.return_attention_weights:
                out = self.attention(query, out, **kwargs)  # (batch_size, 1, n_time_steps, lstm_units)
            else:
                out, attention_weights = self.attention(query, out, return_attention_scores=True, **kwargs)
                attention_weights = tf.squeeze(attention_weights, axis=2)

            # Remove the extra dimension (batch_size, 1, n_time_steps, lstm_units)
            out = tf.squeeze(out, axis=1)
            # pooling over time steps
            out = self.pooling(out, **kwargs)  # (batch_size, lstm_units)
        else:  # group attention
            # if no time-attention is used, padded time series is masked
            # assuming that all zero values are the padding
            mask = tf.reduce_any(tf.not_equal(x, 0), axis=-1)  # Shape: (batch_size, n_groups, n_time_steps)
            out = self.group_rnn(x, mask=mask, **kwargs)  # (batch_size, n_groups, lstm_units)
            # bidirectional LSTM returns 2*lstm_units

            # Reshape for group-based multi-head attention
            query = tf.reduce_mean(out, axis=1, keepdims=True)  # Shape: (batch_size, 1, lstm_units)

            # Apply group-based attention
            if not self.return_attention_weights:
                out = self.attention(query, out, **kwargs)
            else:
                out, attention_weights = self.attention(query, out, return_attention_scores=True, **kwargs)
                attention_weights = tf.squeeze(attention_weights, axis=2)

            # Remove the extra dimension (batch_size, 1, lstm_units)
            out = tf.squeeze(out, axis=1)

        # apply layer normalization
        out = self.norm_layer(out)

        # apply dense layer
        out = self.out_layer(out, **kwargs)  # (batch_size, summary_dim)

        if self.return_attention_weights:
            return out, attention_weights
        return out


class EnsembleTrainer:
    """
    Ensemble of trainers to load multiple trained amortizers with different configurations for joint prediction.
    """
    def __init__(self, trainers):
        self.trainers = trainers
        self.n_trainers = len(trainers)
        self.checkpoint_path = 'amortizer_ensemble'
        self.amortizer = self.EnsembleAmortizer([trainer.amortizer for trainer in trainers])
        self.loss_history = self.EnsembleLossHistory(trainers)

    def configurator(self, forward_dict: dict) -> list[dict]:
        out_list = []
        for trainer in self.trainers:
            out = trainer.configurator(forward_dict)
            out_list.append(out)
        return out_list

    class EnsembleAmortizer:
        def __init__(self, amortizers):
            self.amortizers = amortizers
            self.n_amortizers = len(amortizers)

        def sample(self, forward_dict: list[dict], n_samples: int) -> np.ndarray:
            if self.n_amortizers != len(forward_dict):
                raise ValueError(f'Number of forward_dicts ({len(forward_dict)})'
                                 f' does not match number of amortizers ({self.n_amortizers}).')

            out_list = []
            n_samples_per_amortizer = np.ones(self.n_amortizers) * (n_samples // self.n_amortizers)
            n_samples_per_amortizer[:n_samples % self.n_amortizers] += 1

            for a_i, amortizer in enumerate(self.amortizers):
                out = amortizer.sample(forward_dict[a_i], n_samples=n_samples_per_amortizer[a_i])
                out_list.append(out)
            if out_list[0].ndim == 2:
                return np.concatenate(out_list, axis=0)
            return np.concatenate(out_list, axis=1)

    class EnsembleLossHistory:
        def __init__(self, trainers):
            self.trainers = trainers

        def get_plottable(self):
            # Collect all DataFrames for each trainer's train and validation losses
            train_dfs = []
            val_dfs = []

            for trainer in self.trainers:
                history = trainer.loss_history.get_plottable()
                train_dfs.append(history['train_losses'])
                val_dfs.append(history['val_losses'])

            # Calculate the average DataFrame across trainers for both train and val losses
            avg_train_df = pd.concat(train_dfs).groupby(level=0).mean()
            avg_val_df = pd.concat(val_dfs).groupby(level=0).mean()

            return {
                'train_losses': avg_train_df,
                'val_losses': avg_val_df
            }


def load_model(model_id: int, n_params: int, generative_model,
               prior_mean: np.ndarray, prior_std: np.ndarray,
               train_network: bool, valid_data: dict,
               presim_folder: str, amortizer_folder: str,
               make_prior_relative: callable = None,
               amortizer_return_attention_weights: bool = False,
               keep_selection_inference: str = None,
               drop_n_households: Union[bool, float] = False):
    """
    Load or train an amortizer model with a specific configuration.

    :param model_id: model configuration id
    :param n_params: number of parameters
    :param generative_model: the generative model
    :param prior_mean: the prior mean
    :param prior_std: the prior standard deviation
    :param train_network: whether to train the network
    :param valid_data: validation data for training
    :param presim_folder: folder with pre-simulated data
    :param amortizer_folder: folder to save the amortizer
    :param make_prior_relative: function to make the prior relative
    :param amortizer_return_attention_weights: whether to return attention weights (to diagnose attention)
    :param keep_selection_inference: selection procedure to keep during inference (same networks are only trained on
        specific selection procedures)
    :param drop_n_households: whether to drop a random number of households or a specific fraction
    :return:
    """
    iterations_per_epoch = 1000
    # 10000 batches to be generated, 10 epoch until batches are used up
    max_epochs = 300

    coupling_settings_spline = {
        "num_dense": 3,
        "dense_args": dict(
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        "dropout_prob": 0.2,
        "bins": 16,
    }
    summary_loss = None

    fixed_selection = [None, 'pedcov', 'random']
    use_time_attention = [True, False]
    num_coupling_layers = [6, 7, 8, 9]

    net_configs = list(product(fixed_selection, use_time_attention, num_coupling_layers))
    if model_id < 0:
        # load ensemble model of all configurations which should be unbiased
        # only load time attention models, other models have worse loss
        if model_id == -1:  # unbiased on pedcov
            model_ids = [i for i in range(len(net_configs)) if net_configs[i][0] != 'random' and net_configs[i][1]]
            logger.info(f"Load ensemble of all configurations which are unbiased on PedCov")
        elif model_id == -2:  # unbiased on random
            model_ids = [i for i in range(len(net_configs)) if net_configs[i][0] != 'pedcov' and net_configs[i][1]]
            logger.info(f"Load ensemble of all configurations which are unbiased on Random")
        elif model_id == -3:  # unbiased on pedcov and random
            model_ids = [i for i in range(len(net_configs)) if net_configs[i][0] is None and net_configs[i][1]]
            logger.info(f"Load ensemble of all configurations which are unbiased on PedCov and Random")
        elif model_id == -4:  # biased towards pedcov
            model_ids = [i for i in range(len(net_configs)) if net_configs[i][0] == 'pedcov' and net_configs[i][1]]
            logger.info(f"Load ensemble of all configurations which are biased towards PedCov")
        elif model_id == -5:  # biased towards random
            model_ids = [i for i in range(len(net_configs)) if net_configs[i][0] == 'random' and net_configs[i][1]]
            logger.info(f"Load ensemble of all configurations which are biased towards Random")
        else:
            raise ValueError(f"Model ID {model_id} is not valid for ensemble training. "
                             f"Choose a number between -3 and -1")
        trainers = []
        for m_id in model_ids:
            trainer, _ = load_model(
                model_id=m_id,
                n_params=n_params,
                generative_model=generative_model,
                prior_mean=prior_mean,
                prior_std=prior_std,
                train_network=train_network,
                valid_data=valid_data,
                presim_folder=presim_folder,
                amortizer_folder=amortizer_folder,
                make_prior_relative=make_prior_relative,
                amortizer_return_attention_weights=amortizer_return_attention_weights,
                keep_selection_inference=keep_selection_inference,
                drop_n_households=drop_n_households
            )
            trainers.append(trainer)
        return EnsembleTrainer(trainers), 'amortizer_ensemble'

    elif model_id >= len(net_configs):
        raise ValueError(f"Model ID {model_id} is out of range. Choose a number between 0 and {len(net_configs) - 1}")

    fixed_selection, use_time_attention, num_coupling_layers = list(net_configs)[model_id]
    amortizer_name = (f"amortizer_{model_id}"
                      f"{'-'+fixed_selection+'_only' if fixed_selection is not None else ''}"
                      f"{'-time_attention' if use_time_attention else '_group_attention'}"
                      f"-{num_coupling_layers}_layers"
                      f"{'-drop_households' if drop_n_households else ''}")
    logger.info(f" Model Configuration: {amortizer_name}")

    summary_net = GroupSummaryNetwork(
        summary_dim=n_params * 2,
        use_time_attention=use_time_attention,
        return_attention_weights=amortizer_return_attention_weights,  # only needed for diagnostics of attention
    )

    inference_net = InvertibleNetwork(
        num_params=n_params,
        num_coupling_layers=num_coupling_layers,
        coupling_design='spline',
        coupling_settings=coupling_settings_spline,
        permutation="learnable"
    )

    amortizer = AmortizedPosterior(
        inference_net=inference_net,
        summary_net=summary_net,
        summary_loss_fun=summary_loss
    )

    checkpoint_path = amortizer_folder + '/' + amortizer_name
    os.makedirs(amortizer_folder, exist_ok=True)
    logger.info(f'Checkpoint path: {amortizer_folder}/{amortizer_name} ')

    # build the trainer with networks and generative model
    max_to_keep = 7
    if make_prior_relative is None:  # no joint inference
        trainer = Trainer(
            amortizer=amortizer,
            # during training, we drop random selection if we want to train on PedCov only
            # during inference, we are not dropping random selection, but we do not inform the network about the selection procedure
            configurator=partial(configurator, prior_mean=prior_mean, prior_std=prior_std,
                                 keep_selection=fixed_selection,
                                 drop_n_households=drop_n_households) if train_network else
            partial(configurator, prior_mean=prior_mean, prior_std=prior_std,
                    keep_selection=keep_selection_inference, not_inform_selection=fixed_selection,
                    drop_n_households=drop_n_households),
            generative_model=generative_model,
            checkpoint_path=checkpoint_path,
            skip_checks=True,
            max_to_keep=max_to_keep
        )
    else:
        trainer = Trainer(
            amortizer=amortizer,
            # during training, we drop random selection if we want to train on PedCov only
            # during inference, we are not dropping random selection, but we do not inform the network about the selection procedure
            configurator=partial(configurator_joint, prior_mean=prior_mean, prior_std=prior_std,
                                 make_prior_relative=make_prior_relative,
                                 keep_selection=fixed_selection) if train_network else
            partial(configurator_joint, prior_mean=prior_mean, prior_std=prior_std,
                    make_prior_relative=make_prior_relative,
                    keep_selection=keep_selection_inference, not_inform_selection=fixed_selection),
            generative_model=generative_model,
            checkpoint_path=checkpoint_path,
            skip_checks=True,
            max_to_keep=max_to_keep
        )

    if train_network:
        # simulation done before, start training now
        trainer._setup_optimizer(
            optimizer=None,
            epochs=max_epochs,
            iterations_per_epoch=iterations_per_epoch
        )
        optimizer = trainer.optimizer

        start_time = perf_counter()
        history = trainer.train_from_presimulation(
            presimulation_path=presim_folder,
            optimizer=optimizer,
            max_epochs=max_epochs,
            early_stopping=True,
            custom_loader=CustomLoader(file_path=presim_folder, iterations_per_epoch=iterations_per_epoch),
            validation_sims=valid_data
        )

        end_time = perf_counter()
        logger.info(f'training time: {(end_time - start_time) / 60} minutes')
    else:
        trainer.load_pretrained_network()
        history = trainer.loss_history.get_plottable()

    if 'val_losses' in history.keys():
        # Check if training converged
        if np.isnan(history['val_losses'].iloc[-1]).any():
            logger.warning('Training failed with NaN loss at the end')
            if np.isnan(history['val_losses'].iloc[-max_to_keep:]).all():
                logger.warning('Training failed with NaN loss for all latest checkpoints')

        # Find the checkpoint with the lowest validation loss out of the last 7
        recent_losses = history['val_losses'].iloc[-max_to_keep:]
        best_valid_epoch = recent_losses['Loss'].idxmin() + 1  # checkpoints are 1-based indexed
        new_checkpoint = trainer.manager.latest_checkpoint.rsplit('-', 1)[0] + f'-{best_valid_epoch}'
        trainer.checkpoint.restore(new_checkpoint)
        logger.info(f"Networks loaded from {new_checkpoint}")
        logger.info(f"Best validation loss at epoch {best_valid_epoch} with {recent_losses['Loss'][best_valid_epoch-1]}")
    else:
        logger.warning('No validation losses found in history')

    return trainer, amortizer_name

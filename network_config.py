import os
import pickle
from functools import partial
from time import perf_counter
from typing import Union

import numpy as np
import tensorflow as tf
from bayesflow import default_settings as defaults
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.helper_networks import MultiConv1D
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional
from tensorflow.keras.models import Sequential


def custom_loader(file_path):
    """Uses pickle to load, but each path is folder with multiple files, each one batch"""
    # load all files in folder
    loaded_presimulations = []
    for file in os.listdir(file_path):
        with open(os.path.join(file_path, file), 'rb') as batch_file:
            batch = pickle.load(batch_file)[0]
            assert isinstance(batch, dict)  # only one batch per file
            loaded_presimulations.append(batch)
    # shuffle list, so iterations are random, only batches stay the same
    np.random.shuffle(loaded_presimulations)
    return loaded_presimulations


def configurator(forward_dict: dict,
                 prior_mean: np.ndarray, prior_std: np.ndarray,
                 not_inform_selection: bool = False, drop_random: bool = False, drop_n_households = False) -> dict:
    out_dict = {}

    # Extract data (already normalized)
    x = forward_dict["sim_data"]
    if drop_n_households:
        if isinstance(drop_n_households, bool):
            drop_n_households = np.random.choice([0., 0.1, 0.3, 0.5, 0.7, 0.9])
        drop_n_households = int(drop_n_households * x.shape[1])
        print(f'Warning: Dropping {drop_n_households} / {x.shape[1]} households')
        x = x[:, drop_n_households:]
    out_dict['summary_conditions'] = x.astype(np.float32)

    # Extract params
    if 'parameters' in forward_dict.keys():
        forward_dict["prior_draws"] = forward_dict["parameters"]
    if 'prior_draws' in forward_dict.keys():
        params = forward_dict["prior_draws"]
        # normalize
        params = (params - prior_mean) / prior_std
        out_dict['parameters'] = params.astype(np.float32)

    # non batchable context can come as a list or as a single value
    if isinstance(forward_dict['sim_non_batchable_context'], str):
        forward_dict['sim_non_batchable_context'] = [forward_dict['sim_non_batchable_context']] * len \
            (forward_dict['sim_data'])
    sim_non_batchable_context = np.array(forward_dict['sim_non_batchable_context']).flatten()
    sim_batchable_context = np.array(forward_dict['sim_batchable_context']).flatten()

    # Extract context
    variant_selection = np.array(
        [[0. if v == 'alpha' else 1., 0. if s == 'pedcov' else 1.]
         for v, s in zip(sim_non_batchable_context, sim_batchable_context)]
    )

    if drop_random:
        print('Warning: Dropping random selection')
        not_inform_selection = True
        # Extract context
        keep_indices = np.array([True if s == 'pedcov' else False for s in sim_batchable_context])
        if keep_indices.size == 0:
            # If keep_indices is empty, select a random index to not have an empty batch
            print("keep_indices is empty, select a random index")
            keep_indices = np.array([np.random.choice(len(sim_batchable_context))])

        variant_selection = variant_selection[keep_indices]
        out_dict['summary_conditions'] = out_dict['summary_conditions'][keep_indices]
        if 'parameters' in out_dict.keys():
            out_dict['parameters'] = out_dict['parameters'][keep_indices]

    if not_inform_selection:
        # drop second direct condition on selection procedure
        print('Warning: Not informing about selection procedure')
        variant_selection = variant_selection[:, 0][:, np.newaxis]

    out_dict['direct_condition'] = variant_selection.astype(np.float32)

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
                       not_inform_selection: bool = False, drop_random: bool = False) -> dict:
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

    # non batchable context can come as a list or as a single value
    sim_batchable_context = np.array(forward_dict['sim_batchable_context']).flatten()

    # Extract context
    selection_cond = np.array(
        [[0. if s == 'pedcov' else 1.]
         for s in sim_batchable_context]
    )

    if drop_random:
        print('Warning: Dropping random selection')
        not_inform_selection = True
        # Extract context
        keep_indices = np.array([True if s == 'pedcov' else False for s in sim_batchable_context])
        if keep_indices.size == 0:
            # If keep_indices is empty, select a random index to not have an empty batch
            print("keep_indices is empty, select a random index")
            keep_indices = np.array([np.random.choice(len(sim_batchable_context))])

        selection_cond = selection_cond[keep_indices]
        out_dict['summary_conditions'] = out_dict['summary_conditions'][keep_indices]
        if 'parameters' in out_dict.keys():
            out_dict['parameters'] = out_dict['parameters'][keep_indices]

    if not_inform_selection:
        # drop direct condition on selection procedure
        print('Warning: Not informing about selection procedure')
    else:
        out_dict['direct_condition'] = selection_cond.astype(np.float32)

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
            num_conv_layers=2,
            rnn_units=128,
            bidirectional=True,
            conv_settings=None,
            use_attention=False,
            return_attention_weights=False,
            use_GRU=True,
            **kwargs
    ):
        super().__init__(**kwargs)

        if conv_settings is None:
            conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV

        conv = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])
        self.group_conv = tf.keras.layers.TimeDistributed(conv)
        self.use_attention = use_attention
        self.return_attention_weights = return_attention_weights
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        if self.use_attention:
            self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=rnn_units)

        if use_GRU:
            rnn = Bidirectional(GRU(rnn_units, return_sequences=use_attention)) if bidirectional else GRU(rnn_units,
                                                                                                          return_sequences=use_attention)
        else:
            rnn = Bidirectional(LSTM(rnn_units, return_sequences=use_attention)) if bidirectional else LSTM(rnn_units,
                                                                                                            return_sequences=use_attention)
        self.group_rnn = tf.keras.layers.TimeDistributed(rnn)

        self.out_layer = Dense(summary_dim, activation="linear")
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
        # Apply the RNN to each group
        out = self.group_conv(x, **kwargs)
        out = self.group_rnn(out, **kwargs)  # (batch_size, n_groups, lstm_units)
        # if attention is used, return full sequence (batch_size, n_groups, n_time_steps, lstm_units)
        # bidirectional LSTM returns 2*lstm_units

        if self.use_attention:
            # learn a query vector to attend over the time points
            query = tf.reduce_mean(out, axis=1)
            # Reshape query to match the required shape for attention
            query = tf.expand_dims(query, axis=1)  # (batch_size, 1, n_time_steps, lstm_units)
            if not self.return_attention_weights:
                out = self.attention(query, out, **kwargs)  # (batch_size, 1, n_time_steps, lstm_units)
            else:
                out, attention_weights = self.attention(query, out, return_attention_scores=True, **kwargs)
                attention_weights = tf.squeeze(attention_weights, axis=2)
            out = tf.squeeze(out, axis=1)  # Remove the extra dimension (batch_size, n_time_steps, lstm_units)
            out = self.pooling(out, **kwargs)  # (batch_size, 1, lstm_units)
        else:
            # pooling over groups, this totally invariants to the order of the groups
            out = self.pooling(out, **kwargs)  # (batch_size, lstm_units)
        # apply dense layer
        out = self.out_layer(out, **kwargs)  # (batch_size, summary_dim)

        if self.return_attention_weights:
            return out, attention_weights
        return out

def load_model(model_id: int, n_params: int, generative_model,
               prior_mean: np.ndarray, prior_std: np.ndarray,
               train_network: bool, valid_data: dict,
               presim_folder: str, amortizer_folder: str,
               make_prior_relative: callable = None,
               amortizer_return_attention_weights: bool = False,
               drop_n_households: Union[bool, float] = False):
    iterations_per_epoch = 1000
    # 10000 batches to be generated, 10 epoch until batches are used up
    max_epochs = 250  # 100

    coupling_settings_spline = {
        "num_dense": 3,
        "dense_args": dict(
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        ),
        "dropout_prob": 0.2,
        "bins": 16,
    }
    summary_loss = 'MMD'
    drop_random_selection = False

    if model_id == 0:
        amortizer_name = 'amortizer-sampling-bias-both_variants-6'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 6
        use_attention = False
    elif model_id == 1:
        amortizer_name = 'amortizer-sampling-bias-both_variants-6-attention'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 6
        use_attention = True
    elif model_id == 2:
        amortizer_name = 'amortizer-sampling-bias-both_variants-7'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 7
        use_attention = False
    elif model_id == 3:
        amortizer_name = 'amortizer-sampling-bias-both_variants-7-attention'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 7
        use_attention = True
    elif model_id == 4:
        amortizer_name = 'amortizer-sampling-bias-both_variants-8'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 8
        use_attention = False
    elif model_id == 5:
        amortizer_name = 'amortizer-sampling-bias-both_variants-8-attention'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 8
        use_attention = True
    elif model_id == 6:
        amortizer_name = 'amortizer-sampling-bias-both_variants-9-attention'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 9
        use_attention = True
    elif model_id == 7:
        amortizer_name = 'amortizer-sampling-bias-both_variants-6-attention-pedcov-only'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 6
        use_attention = True
        drop_random_selection = True
    elif model_id == 8:
        amortizer_name = 'amortizer-sampling-bias-both_variants-7-attention-pedcov-only'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 7
        use_attention = True
        drop_random_selection = True
    elif model_id == 9:
        amortizer_name = 'amortizer-sampling-bias-both_variants-8-attention-pedcov-only'
        coupling_settings = coupling_settings_spline
        num_coupling_layers = 8
        use_attention = True
        drop_random_selection = True
    else:
        raise ValueError('Invalid network id')

    summary_net = GroupSummaryNetwork(
        summary_dim=n_params * 2,
        rnn_units=32,  # multiple of 32 hidden units
        use_attention=use_attention,
        return_attention_weights=amortizer_return_attention_weights,  # only needed for diagnostics of attention
    )

    inference_net = InvertibleNetwork(
        num_params=n_params,
        num_coupling_layers=num_coupling_layers,
        coupling_design='spline',
        coupling_settings=coupling_settings,
        permutation="learnable"
    )

    amortizer = AmortizedPosterior(
        inference_net=inference_net,
        summary_net=summary_net,
        summary_loss_fun=summary_loss
    )

    checkpoint_path = amortizer_folder + '/' + amortizer_name
    os.makedirs(amortizer_folder, exist_ok=True)
    print(checkpoint_path)

    # build the trainer with networks and generative model
    max_to_keep = 7
    if make_prior_relative is None:  # no joint inference
        trainer = Trainer(
            amortizer=amortizer,
            # during training, we drop random selection if we want to train on PedCov only
            # during inference, we are not dropping random selection, but we do not inform the network about the selection procedure
            configurator=partial(configurator, prior_mean=prior_mean, prior_std=prior_std,
                                 drop_random=drop_random_selection, drop_n_households=drop_n_households) if train_network else
            partial(configurator, prior_mean=prior_mean, prior_std=prior_std,
                    not_inform_selection=drop_random_selection, drop_n_households=drop_n_households),
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
                                 drop_random=drop_random_selection) if train_network else
            partial(configurator_joint, prior_mean=prior_mean, prior_std=prior_std,
                    make_prior_relative=make_prior_relative,
                    not_inform_selection=drop_random_selection),
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
            custom_loader=custom_loader,
            validation_sims=valid_data
        )

        end_time = perf_counter()
        print(f'training time: {(end_time - start_time) / 60} minutes')
    else:
        trainer.load_pretrained_network()
        history = trainer.loss_history.get_plottable()

    if 'val_losses' in history.keys():
        # Check if training converged
        if np.isnan(history['val_losses'].iloc[-1]).any():
            print('Training failed with NaN loss at the end')
            if np.isnan(history['val_losses'].iloc[-max_to_keep:]).all():
                print('Training failed with NaN loss for all latest checkpoints')

        # Find the checkpoint with the lowest validation loss out of the last 7
        recent_losses = history['val_losses'].iloc[-max_to_keep:]
        best_valid_epoch = recent_losses['Loss'].idxmin() + 1  # checkpoints are 1-based indexed
        new_checkpoint = trainer.manager.latest_checkpoint.rsplit('-', 1)[0] + f'-{best_valid_epoch}'
        trainer.checkpoint.restore(new_checkpoint)
        print("Networks loaded from {}".format(new_checkpoint))
        print(f"Best validation loss: {recent_losses['Loss'].min()}")
    else:
        print('No validation losses found, using latest checkpoint')

    return trainer, amortizer_name

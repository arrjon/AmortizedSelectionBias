from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from rpy2.robjects import conversion, default_converter, numpy2ri, pandas2ri, ListVector


def dict_to_named_list(dct):
    # function taken from pyabc
    if (
            isinstance(dct, dict)
            or isinstance(dct, pd.core.series.Series)
    ):
        dct = dict(dct.items())
        # convert numbers, numpy arrays and pandas dataframes to builtin
        # types before conversion (see rpy2 #548)
        with conversion.localconverter(
                default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            for key, val in dct.items():
                dct[key] = conversion.py2rpy(val)
        r_list = ListVector(dct)
        return r_list
    return dct


# one-hot-encoding
DICT_ENCODING = {
    'infect_status': {
        0: [0, 0],  # not infected
        1: [1, 0],  # infected and symptomatic
        2: [0, 1],  # infected and asymptomatic
    },
    'age': {
        0: [0, 0],  # <6 years old, I
        1: [1, 0],  # 6-11 years old, C
        2: [0, 1],  # >11 years old, A
    },
    'protected': {
        0: [0],  # not protected
        1: [1]  # protected
    }
}

DATE_MAX = 1000  # maximum date in the dataset


def encode_row(row):
    encoded = [row['date_sympt_norm']]
    for column, encoding in DICT_ENCODING.items():
        encoded.extend(encoding[row[column]])

    return encoded


def normalize_household_data(
        df: pd.DataFrame,
        minimal_length: int,
        use_one_hot_encoding: bool = True
) -> Union[np.ndarray, list]:
    """
    Normalizes the household data and returns it as a numpy array or list.
    Patients in a household are order by event date.

    Parameters
    ----------
    df : pd.DataFrame - the household data
    minimal_length : int - the minimal length of the household data: in long format, this is the total sequence length,
                    in list format, this is the minimal number of people in a household. In both cases, we use zero
                    padding at the beginning of the sequence to reach this length.
    use_one_hot_encoding : bool - if True, uses one-hot encoding for the features
    """
    df = df.copy()
    all_households = []

    # date_sympt = 1000 is not infected
    df.loc[df['date_sympt'] != 1000, 'date_sympt_norm'] = df.loc[df['date_sympt'] != 1000, 'date_sympt'] / DATE_MAX
    df['end_followup_norm'] = df['end_followup'] / DATE_MAX
    #df.loc[df['date_sympt'] == 1000, 'date_sympt_norm'] = 1  # when sorting it stays at the end
    df.loc[df['date_sympt'] == 1000, 'date_sympt_norm'] = df.loc[df['date_sympt'] == 1000, 'end_followup_norm']

    unique_households = df['id_hh'].unique()

    for i, id_hh in enumerate(unique_households):
        df_hh = df[df['id_hh'] == id_hh]
        if use_one_hot_encoding:
            # One-hot encode specific columns
            encoded_data = df_hh.apply(lambda row: encode_row(row), axis=1)
            encoded_household = np.array(encoded_data.tolist())
            follow_up = -np.ones((1, encoded_household.shape[1]))
            follow_up[0, 0] = df_hh['end_followup_norm'].values[0]

            household = np.concatenate((
                encoded_household,
                follow_up
            ), axis=0).T
        else:
            # household as a list
            household = np.stack((
                df_hh['date_sympt_norm'].values,
                df_hh['infect_status'].values,
                df_hh['age'].values,
                df_hh['protected'].values
            ))
            household = np.concatenate((
                household,
                [[df_hh['end_followup_norm'].values[0]],
                 [-1], [-1], [-1]]
            ), axis=1)

        # there is no specific order in the household, so order by date of symptoms
        order = np.argsort(household[0])
        household = household[:, order]

        if household.shape[1] < minimal_length:
            # pad sequence length with zeros, each household gets padded individually
            household = np.concatenate([np.zeros((household.shape[0],
                                                  minimal_length - household.shape[1])),
                                        household], axis=1)
        all_households.append(household)

    if minimal_length > 0:
        return np.stack([h.T for h in all_households])
    # households are returned as list since might have different lengths
    return all_households


def shorten_follow_up_time(data: np.ndarray, max_followup: int) -> np.ndarray:
    """
    Shorten the follow-up time to a given maximum follow-up time. Input can be a single simulation or multiple.
    """
    max_followup = max_followup / DATE_MAX  # normalise the same way as the dates

    if data.ndim == 3:
        # only one simulation with multiple households
        household_data = data.copy()
        sim_data_list = household_data[np.newaxis]
    elif data.ndim == 4:
        # multiple simulations, each with multiple households
        sim_data_list = data.copy()
    else:
        raise ValueError(f"The data must have 3 or 4 dimensions, but has {data.ndim} dimensions.")

    for s_i, household_data in enumerate(sim_data_list):
        for household in household_data:
            # infection status is not known after the end of follow-up
            unobserved_members = household[:, 0] > max_followup
            household[:, 0][unobserved_members] = max_followup
            household[:, 1][unobserved_members] = 0
            household[-1, 1] = -1  # end of follow-up
        sim_data_list[s_i] = household_data
    if data.ndim == 3:
        return sim_data_list[0]
    return sim_data_list


def get_household_statistic(data: np.ndarray) -> dict:
    """
    Get the number of infections by household size, age group, and infection status.
    """
    from collections import defaultdict

    # Initialize counters
    infection_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    time_distribution = defaultdict(lambda: defaultdict(list))
    time_distribution_age = defaultdict(lambda: defaultdict(list))

    # Iterate through each household
    for household in data:
        household_members = household[:, 0] != 0
        household = household[household_members][:-1]  # remove follow-up date
        household_size = household.shape[0]
        infection_times = household[:, 0]
        infection_statuses = household[:, 1:3]
        infection_statuses = np.where(infection_statuses.sum(axis=1) == 0, 0, infection_statuses.argmax(axis=1) + 1)
        age_group = household[:, 3:5]
        age_group = np.where(age_group.sum(axis=1) == 0, 0, age_group.argmax(axis=1) + 1)
        protection_status = household[:, 5]

        for i in range(household_size):
            # Update the count based on household size, age group, and protection status
            infection_counts[household_size][age_group[i]][infection_statuses[i]] += 1

            # Record the infection time for distribution analysis
            time_distribution[household_size][infection_statuses[i]].append(infection_times[i])
            time_distribution_age[household_size][age_group[i]].append(infection_times[i])

    infection_counts = pd.DataFrame(infection_counts)
    time_distribution = pd.DataFrame(time_distribution)
    time_distribution_age = pd.DataFrame(time_distribution_age)

    # change index
    age_index = ['<6 years', '6-11 years', '>11 years']
    status_index = ['not_infected', 'infected_symptomatic', 'infected_asymptomatic']
    infection_counts.index = [age_index[i] for i in infection_counts.index]
    infection_counts = infection_counts.reindex(age_index)  # reorder

    time_distribution.index = [status_index[i] for i in time_distribution.index]
    time_distribution = time_distribution.reindex(status_index)  # reorder
    time_distribution_age.index = [age_index[i] for i in time_distribution_age.index]
    time_distribution_age = time_distribution_age.reindex(age_index)  # reorder

    out_dict = {
        'infection_counts': infection_counts,
        'time_distribution': time_distribution,
        'time_distribution_age': time_distribution_age,
        'household_sizes': sorted(time_distribution_age.columns)
    }
    return out_dict


def measure_bias(true_values, estimated_values, param_names):
    # Ensure the arrays have the correct shape
    if true_values.shape != estimated_values.shape:
        raise ValueError("The shape of true values and estimated values must be the same.")

    n_reps, n_params = true_values.shape

    results = []

    for i in range(n_params):
        if param_names[i].startswith('mu'):
            true = np.exp(true_values[:, i])
            est = np.exp(estimated_values[:, i])
        else:
            true = true_values[:, i]
            est = estimated_values[:, i]

        bias = est - true
        percent_bias = np.where(true != 0, (bias / true) * 100, np.inf)
        relative_bias = np.where(true != 0, bias / true, np.inf)
        absolute_error = np.abs(bias)

        results.append({
            'Parameter': param_names[i],
            'Mean Bias': np.mean(bias),
            'Mean Percent Bias': np.mean(percent_bias),
            'Mean Relative Bias': np.mean(relative_bias),
            'Mean Absolute Error': np.mean(absolute_error),
            'RMSE': np.sqrt(np.mean(bias ** 2))
        })

    return pd.DataFrame(results)


def plot_attention_scores(
        attention_scores: np.ndarray,
        valid_data: Optional[np.ndarray] = None,
        batch_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        group_idx: Optional[int] = None,
        normalize: bool = True
):
    """
    Plots a heatmap of attention scores for a specific batch, head or group.

    Parameters:
    -----------
    attention_scores : tf.Tensor
        Attention scores tensor of shape (batch_size, num_heads, n_time_steps, n_groups, n_time_steps).
    head_idx : int
        Index of the attention head to visualize.
    group_idx : int
        Index of the group to visualize.
    """
    # Extract the scores for the given batch, head, and group
    if group_idx is None:
        scores = tf.reduce_mean(attention_scores, axis=3)  # Average over groups
    else:
        scores = attention_scores[:, :, :, group_idx, :]
    if batch_idx is None:
        scores = tf.reduce_mean(scores, axis=0)  # average over batches
    else:
        scores = scores[batch_idx, :, :, :]
    if head_idx is None:
        scores = tf.reduce_mean(scores, axis=0).numpy()  # Shape: (9, 9)
    else:
        scores = scores[head_idx, :, :].numpy()  # Shape: (9, 9)

    # Normalize the scores
    if normalize:
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    first_inv_idx, time_labels = None, None
    if valid_data is not None and batch_idx is not None and group_idx is not None:
        time_labels = valid_data['sim_data'][batch_idx, group_idx, :, 0]  # Extract time step label
        first_inv_idx = np.where(valid_data['sim_data'][batch_idx, group_idx, :, 0] != 0)[0][0]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(scores, cmap='viridis', aspect='auto')
    if normalize:
        plt.colorbar(label='Normalized Attention Score')
    else:
        plt.colorbar(label='Attention Score')
    plt.title(f"Attention Scores for, Head {head_idx}")
    plt.xlabel('Key/Value Time Steps')
    plt.ylabel('Query Time Steps')
    if first_inv_idx is not None:
        # Add a vertical line at the index of first infected
        plt.vlines(x=first_inv_idx - 0.5, ymin=first_inv_idx - 0.5, ymax=scores.shape[0] - 0.5,
                   color='red', linestyle='--', label='First 1')
        plt.hlines(y=first_inv_idx - 0.5, xmin=first_inv_idx - 0.5, xmax=scores.shape[0] - 0.5,
                   color='red', linestyle='--', label='First 1')

    # Set time step labels if available
    if time_labels is not None:
        plt.xticks(ticks=np.arange(len(time_labels)), labels=time_labels, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(time_labels)), labels=time_labels)

    plt.show()


def plot_attention_scores_plotly(
        attention_scores: np.ndarray,
        valid_data: Optional[np.ndarray] = None,
        batch_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        group_idx: Optional[int] = None,
        normalize: bool = True
):
    """
    Plots a heatmap of attention scores for a specific batch, head, or group using Plotly.

    Parameters:
    -----------
    attention_scores : np.ndarray
        Attention scores tensor of shape (batch_size, num_heads, n_time_steps, n_groups, n_time_steps).
    valid_data : np.ndarray
        Optional data containing time step information.
    batch_idx : int
        Index of the batch to visualize.
    head_idx : int
        Index of the attention head to visualize.
    group_idx : int
        Index of the group to visualize.
    normalize : bool
        Whether to normalize the attention scores.
    """
    import plotly.graph_objects as go

    # Extract the scores for the given batch, head, and group
    if group_idx is None:
        scores = np.mean(attention_scores, axis=3)  # Average over groups
    else:
        scores = attention_scores[:, :, :, group_idx, :]
    if batch_idx is None:
        scores = np.mean(scores, axis=0)  # Average over batches
    else:
        scores = scores[batch_idx, :, :, :]
    if head_idx is None:
        scores = np.mean(scores, axis=0)  # Shape: (n_time_steps, n_time_steps)
    else:
        scores = scores[head_idx, :, :]  # Shape: (n_time_steps, n_time_steps)

    # Normalize the scores
    if normalize:
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    time_labels = None
    if valid_data is not None and batch_idx is not None and group_idx is not None:
        # Extract time labels and positions
        time_labels = valid_data['sim_data'][batch_idx, group_idx, :, 0]  # Extract time step label
        non_zero_indices = time_labels != 0
        scores = scores[non_zero_indices, :][:, non_zero_indices]
        time_labels = time_labels[non_zero_indices]

    # Create edges for x and y if time positions are given
    if time_labels is not None:
        # Create edges for the heatmap blocks
        x_edges = time_labels
        y_edges = time_labels
    else:
        # Use default positions if time_positions is not provided
        n_time_steps = scores.shape[0]
        x_edges = np.arange(n_time_steps + 1)
        y_edges = np.arange(n_time_steps + 1)

    # Create the heatmap using Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=scores,
            x=x_edges,
            y=y_edges,
            colorscale='Viridis',
            colorbar=dict(title='Normalized Attention Score' if normalize else 'Attention Score')
        )
    )

    # Set the layout with labels and titles
    fig.update_layout(
        title=f"Attention Scores for Head {head_idx}",
        xaxis_title='Key/Value Time Steps',
        yaxis_title='Query Time Steps',
        xaxis=dict(tickmode='array', tickvals=(x_edges[:-1] + np.diff(x_edges) / 2), ticktext=time_labels, tickangle=90),
        yaxis=dict(tickmode='array', tickvals=(y_edges[:-1] + np.diff(y_edges) / 2), ticktext=time_labels, autorange='reversed'),
        width=800,
        height=600
    )
    fig.show()

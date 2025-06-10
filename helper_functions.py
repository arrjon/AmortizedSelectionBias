from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
import pandas as pd


class InfectionStatus(IntEnum):
    """Enumeration for infection status values"""
    NOT_INFECTED = 0
    INFECTED_SYMPTOMATIC = 1
    INFECTED_ASYMPTOMATIC = 2

class AgeGroup(IntEnum):
    """Enumeration for infection status values"""
    INFANT = 0
    CHILD = 1
    OLDER = 2


@dataclass
class ProcessingConfig:
    """Configuration parameters for PedCov processing"""
    DATE_MAX: float = 1000.  # maximum date in the dataset
    AGE_MAX: float = 100.
    USE_ONE_HOT: bool = True
    NOT_INFECTED_DATE: float = 1000.  # special value indicating not infected until end of follow up


# One-hot encoding dictionaries
ENCODING_DICT = {
    'infect_status': {
        InfectionStatus.NOT_INFECTED: [0, 0],
        InfectionStatus.INFECTED_SYMPTOMATIC: [1, 0],
        InfectionStatus.INFECTED_ASYMPTOMATIC: [0, 1],
    },
    'age': {
        AgeGroup.INFANT: [0, 0],
        AgeGroup.CHILD: [1, 0],
        AgeGroup.OLDER: [0, 1],
    },
    'protected': {
        0: [0],  # not protected
        1: [1]  # protected
    }
}


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Validates input DataFrame for required columns and PedCov integrity.

    Args:
        df: Input DataFrame to validate

    Raises:
        ValueError: If required columns are missing or PedCov integrity issues are found
    """
    required_columns = {'id_hh', 'date_sympt', 'infect_status', 'age_exact', 'protected', 'end_followup'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate PedCov types and ranges
    if df['id_hh'].isna().any():
        raise ValueError("id_hh contains missing values")

    # Convert infection status to integers and validate
    valid_statuses = set(InfectionStatus)
    actual_statuses = set(df['infect_status'].astype(int).unique())
    invalid_statuses = actual_statuses - valid_statuses
    if invalid_statuses:
        raise ValueError(f"Invalid infection status values found: {invalid_statuses}")


def encode_row(row: pd.Series, encoding_dict: Dict) -> List[float]:
    """
    Encodes a single row of PedCov using the provided encoding dictionary.

    Args:
        row: DataFrame row to encode
        encoding_dict: Dictionary containing encoding mappings

    Returns:
        List of encoded values
    """
    try:
        encoded = [row['date_sympt_norm']]
        for column, encoding in encoding_dict.items():
            encoded.extend(encoding[row[column]])
        return encoded
    except KeyError as e:
        raise KeyError(f"Missing required column for encoding: {e}")
    except Exception as e:
        raise ValueError(f"Error encoding row: {e}")


def normalize_dates_and_age(
        df: pd.DataFrame,
        config: ProcessingConfig
) -> pd.DataFrame:
    """
    Normalizes dates and age values in the DataFrame.

    Args:
        df: Input DataFrame
        config: Processing configuration

    Returns:
        DataFrame with normalized values
    """
    df = df.copy()

    # Normalize symptomatic dates
    mask_infected = df['date_sympt'] != config.NOT_INFECTED_DATE
    df.loc[mask_infected, 'date_sympt_norm'] = df.loc[mask_infected, 'date_sympt'] / config.DATE_MAX

    # Normalize end followup dates
    df['end_followup_norm'] = df['end_followup'] / config.DATE_MAX

    # For non-infected cases, use end_followup_norm
    df.loc[~mask_infected, 'date_sympt_norm'] = df.loc[~mask_infected, 'end_followup_norm']

    # Normalize age
    df['age_exact_norm'] = df['age_exact'] / config.AGE_MAX

    return df


def process_household(
        df_hh: pd.DataFrame,
        minimal_length: int,
        config: ProcessingConfig
) -> np.ndarray:
    """
    Processes PedCov for a single household.

    Args:
        df_hh: DataFrame containing single household PedCov
        minimal_length: Minimum sequence length required
        config: Processing configuration

    Returns:
        Processed household PedCov as numpy array (n_features, time_steps)
    """
    if config.USE_ONE_HOT:
        encoded_data = df_hh.apply(lambda row: encode_row(row, ENCODING_DICT), axis=1)
        encoded_household = np.array(encoded_data.tolist())

        # Create follow-up array
        follow_up = -np.ones((encoded_household.shape[1] + 1, 1))
        follow_up[0, 0] = df_hh['end_followup_norm'].iloc[0]

        # Construct household array without follow-up
        household = np.concatenate((
            encoded_household[:, :5],  # measurement time, infection status, age group
            df_hh['age_exact_norm'].values[:, np.newaxis],
            encoded_household[:, 5][:, np.newaxis],  # protection status
        ), axis=1)

        # Sort only the main PedCov by date (excluding follow-up)
        order = np.argsort(household[:, 0])
        household = household[order]

        # Add follow-up as the last row after sorting
        household = np.concatenate((household, follow_up.T), axis=0)
        household = household.T  # transpose to have shape (n_features, n_members)

    else:
        # Construct main household PedCov
        household = np.stack((
            df_hh['date_sympt_norm'].values,
            df_hh['infect_status'].values,
            df_hh['age'].values,  # age_group
            df_hh['age_exact_norm'].values,
            df_hh['protected'].values
        ))

        # Sort main PedCov by date
        order = np.argsort(household[0])
        household = household[:, order]

        # Add follow-up PedCov as the last column
        follow_up = np.array([[df_hh['end_followup_norm'].iloc[0]], [-1], [-1], [-1], [-1]])
        household = np.concatenate((household, follow_up), axis=1)

    # Pad if necessary
    if household.shape[1] < minimal_length:
        padding = np.zeros((household.shape[0], minimal_length - household.shape[1]))
        household = np.concatenate([padding, household], axis=1)
    return household


def normalize_household_data(
        df: pd.DataFrame,
        minimal_length: int,
        config: ProcessingConfig = ProcessingConfig()
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Normalizes household PedCov and returns it as a numpy array or list.

    Args:
        df: Input DataFrame containing household PedCov
        minimal_length: Minimum sequence length required
        config: Processing configuration

    Returns:
        Processed household PedCov as either numpy array (n_households x time_steps x n_features)
         or list depending on minimal_length

    Raises:
        ValueError: If input PedCov validation fails
    """
    try:
        # Validate input PedCov
        validate_input_data(df)

        # Normalize dates and age
        df = normalize_dates_and_age(df, config)

        # Process each household
        all_households = []
        for id_hh in df['id_hh'].unique():
            df_hh = df[df['id_hh'] == id_hh]
            household = process_household(df_hh, minimal_length, config)
            all_households.append(household.T)  # switch now to (time_steps x n_features)

        # Stack or return as list based on minimal_length
        if minimal_length > 0:
            return np.stack([h for h in all_households])  # now we get (n_households x time_steps x n_features)
        return all_households

    except Exception as e:
        raise ValueError(f"Error processing household PedCov: {e}")

def shorten_follow_up_time(
        data: np.ndarray,
        max_followup: int,
        config: ProcessingConfig = ProcessingConfig()
) -> np.ndarray:
    """
    Shorten the follow-up time to a given maximum follow-up time. Input can be a single simulation or multiple.
    """
    max_followup = max_followup / config.DATE_MAX  # normalise the same way as the dates

    if data.ndim == 3:
        # only one simulation with multiple households
        household_data = data.copy()
        sim_data_list = household_data[np.newaxis]
    elif data.ndim == 4:
        # multiple simulations, each with multiple households
        sim_data_list = data.copy()
    else:
        raise ValueError(f"The PedCov must have 3 or 4 dimensions, but has {data.ndim} dimensions.")

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


# def plot_attention_scores(
#         attention_scores: np.ndarray,
#         valid_data: Optional[np.ndarray] = None,
#         batch_idx: Optional[int] = None,
#         head_idx: Optional[int] = None,
#         group_idx: Optional[int] = None,
#         normalize: bool = True
# ):
#     """
#     Plots a heatmap of attention scores for a specific batch, head or group.
#
#     Parameters:
#     -----------
#     attention_scores : tf.Tensor
#         Attention scores tensor of shape (batch_size, num_heads, n_time_steps, n_groups, n_time_steps).
#     head_idx : int
#         Index of the attention head to visualize.
#     group_idx : int
#         Index of the group to visualize.
#     """
#
#     if attention_scores.ndim == 5:  # time_attention
#         # Extract the scores for the given batch, head, and group
#         if group_idx is None:
#             scores = tf.reduce_mean(attention_scores, axis=3)  # Average over groups
#         else:
#             scores = attention_scores[:, :, :, group_idx, :]
#         if batch_idx is None:
#             scores = tf.reduce_mean(scores, axis=0)  # average over batches
#         else:
#             scores = scores[batch_idx]
#         if head_idx is None:
#             scores = tf.reduce_mean(scores, axis=0).numpy()  # Shape: (9, 9)
#         else:
#             scores = scores[head_idx].numpy()  # Shape: (9, 9)
#
#         first_inv_idx, time_labels = None, None
#         if valid_data is not None and batch_idx is not None and group_idx is not None:
#             time_labels = valid_data['sim_data'][batch_idx, group_idx, :, 0]  # Extract time step label
#             first_inv_idx = np.where(valid_data['sim_data'][batch_idx, group_idx, :, 0] != 0)[0][0]
#
#         # Normalize the scores
#         if normalize:
#             scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
#
#         # Plot the heatmap
#         plt.figure(figsize=(8, 6))
#         plt.imshow(scores, cmap='viridis', aspect='auto')
#         if normalize:
#             plt.colorbar(label='Normalized Attention Score')
#         else:
#             plt.colorbar(label='Attention Score')
#         plt.title(f"Attention Scores for, Head {head_idx}")
#         plt.xlabel('Key/Value Time Steps')
#         plt.ylabel('Query Time Steps')
#         if first_inv_idx is not None:
#             # Add a vertical line at the index of first infected
#             plt.vlines(x=first_inv_idx - 0.5, ymin=first_inv_idx - 0.5, ymax=scores.shape[0] - 0.5,
#                        color='red', linestyle='--', label='First 1')
#             plt.hlines(y=first_inv_idx - 0.5, xmin=first_inv_idx - 0.5, xmax=scores.shape[0] - 0.5,
#                        color='red', linestyle='--', label='First 1')
#
#         # Set time step labels if available
#         if time_labels is not None:
#             plt.xticks(ticks=np.arange(len(time_labels)), labels=time_labels, rotation=45, ha='right')
#             plt.yticks(ticks=np.arange(len(time_labels)), labels=time_labels)
#
#         plt.show()
#
#     elif attention_scores.ndim == 3:
#         scores = attention_scores.numpy()
#         if batch_idx is not None and head_idx is not None:
#             scores = scores[batch_idx, head_idx]
#         elif batch_idx is not None:
#             scores = scores[batch_idx].T
#         elif head_idx is not None:
#             scores = scores[:, head_idx].T
#         else:
#             scores = np.mean(scores, axis=(0,1))
#
#         # plot scores per group
#         plt.hist(scores)
#         plt.xlabel(f'Attention Scores, Batch {batch_idx}, Head {head_idx}')
#         plt.show()
#
#     else:
#         raise ValueError(f'wrong dimensions of attention scores: {attention_scores.ndim}')
#     return
#
#
# def plot_attention_scores_plotly(
#         attention_scores: np.ndarray,
#         valid_data: Optional[np.ndarray] = None,
#         batch_idx: Optional[int] = None,
#         head_idx: Optional[int] = None,
#         group_idx: Optional[int] = None,
#         normalize: bool = True
# ):
#     """
#     Plots a heatmap of attention scores for a specific batch, head, or group using Plotly.
#
#     Parameters:
#     -----------
#     attention_scores : np.ndarray
#         Attention scores tensor of shape (batch_size, num_heads, n_time_steps, n_groups, n_time_steps).
#     valid_data : np.ndarray
#         Optional PedCov containing time step information.
#     batch_idx : int
#         Index of the batch to visualize.
#     head_idx : int
#         Index of the attention head to visualize.
#     group_idx : int
#         Index of the group to visualize.
#     normalize : bool
#         Whether to normalize the attention scores.
#     """
#     import plotly.graph_objects as go
#
#     # Extract the scores for the given batch, head, and group
#     if group_idx is None:
#         scores = np.mean(attention_scores, axis=3)  # Average over groups
#     else:
#         scores = attention_scores[:, :, :, group_idx, :]
#     if batch_idx is None:
#         scores = np.mean(scores, axis=0)  # Average over batches
#     else:
#         scores = scores[batch_idx, :, :, :]
#     if head_idx is None:
#         scores = np.mean(scores, axis=0)  # Shape: (n_time_steps, n_time_steps)
#     else:
#         scores = scores[head_idx, :, :]  # Shape: (n_time_steps, n_time_steps)
#
#     # Normalize the scores
#     if normalize:
#         scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
#
#     time_labels = None
#     if valid_data is not None and batch_idx is not None and group_idx is not None:
#         # Extract time labels and positions
#         time_labels = valid_data['sim_data'][batch_idx, group_idx, :, 0]  # Extract time step label
#         non_zero_indices = time_labels != 0
#         scores = scores[non_zero_indices, :][:, non_zero_indices]
#         time_labels = time_labels[non_zero_indices]
#
#     # Create edges for x and y if time positions are given
#     if time_labels is not None:
#         # Create edges for the heatmap blocks
#         x_edges = time_labels
#         y_edges = time_labels
#     else:
#         # Use default positions if time_positions is not provided
#         n_time_steps = scores.shape[0]
#         x_edges = np.arange(n_time_steps + 1)
#         y_edges = np.arange(n_time_steps + 1)
#
#     # Create the heatmap using Plotly
#     fig = go.Figure(
#         PedCov=go.Heatmap(
#             z=scores,
#             x=x_edges,
#             y=y_edges,
#             colorscale='Viridis',
#             colorbar=dict(title='Normalized Attention Score' if normalize else 'Attention Score')
#         )
#     )
#
#     # Set the layout with labels and titles
#     fig.update_layout(
#         title=f"Attention Scores for Head {head_idx}",
#         xaxis_title='Key/Value Time Steps',
#         yaxis_title='Query Time Steps',
#         xaxis=dict(tickmode='array', tickvals=(x_edges[:-1] + np.diff(x_edges) / 2), ticktext=time_labels, tickangle=90),
#         yaxis=dict(tickmode='array', tickvals=(y_edges[:-1] + np.diff(y_edges) / 2), ticktext=time_labels, autorange='reversed'),
#         width=800,
#         height=600
#     )
#     fig.show()
#     return
#
#
# def percentage_infection_age(PedCov: np.ndarray, param_name) -> np.ndarray:
#     # infection_type: 0=not infected (default case), 1=symptomatic, 2=asymptomatic
#     # age_group: 0=infants (default case), 1=children, 2=adults
#     if param_name[-1] == 'A':
#         age_group = 2
#     elif param_name[-1] == 'C':
#         age_group = 1
#     else:
#         age_group = 0
#     if param_name[-2] == 'A':
#         infection_types = [2]
#     elif param_name[-2] == 'S':
#         infection_types = [1]
#     else:
#         # parameter is susceptibility
#         infection_types = [1, 2]
#
#     percentages = np.zeros(PedCov.shape[0])
#     for i, replicate in enumerate(PedCov):
#         for infection_type in infection_types:
#             # Extract the infection and age columns
#             time_points = replicate[:, :, 0]  # Time point
#             infection = replicate[:, :, 1:3]  # Infection columns (one-hot encoded with first dropped)
#             age = replicate[:, :, 3:5]  # Age group columns (one-hot encoded with first dropped)
#
#             # only count non zeros rows and remove end of follow up
#             valid_time_mask = (time_points > 0)
#             valid_time_mask[:, -1] = False  # exclude the follow-up time
#
#             # Handle infection_type and age_group for default case (not encoded as 1)
#             if infection_type == 0:
#                 infection_mask = np.all(infection == 0, axis=-1)
#             else:
#                 infection_mask = infection[:, :, infection_type - 1] == 1
#
#             if age_group == 0:
#                 age_mask = np.all(age == 0, axis=-1)
#             else:
#                 age_mask = age[:, :, age_group - 1] == 1
#
#             # Combine masks to get the desired people
#             combined_mask = infection_mask & age_mask & valid_time_mask
#
#             # Calculate percentage
#             total_people = replicate[valid_time_mask].shape[0]
#             selected_people = replicate[combined_mask]
#             percentages[i] += (selected_people.shape[0] / total_people) if total_people > 0 else 0
#     return percentages

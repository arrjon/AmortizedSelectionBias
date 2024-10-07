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
dict_encoding = {
    'infect_status': {
        0: [0, 0],  # not infected
        1: [1, 0],  # infected and symptomatic
        2: [0, 1],  # infected and asymptomatic
    },
    'age': {
        0: [0, 0],  # <6 years old
        1: [1, 0],  # 6-11 years old
        2: [0, 1],  # >11 years old
    },
    'protected': {
        0: [0],  # not protected
        1: [1]  # protected
    }
}


def encode_row(row):
    encoded = [row['date_sympt_norm']]
    for column, encoding in dict_encoding.items():
        encoded.extend(encoding[row[column]])

    return encoded


def normalize_household_data(df: pd.DataFrame,
                             minimal_length: int,
                             use_one_hot_encoding: bool = True) -> Union[np.ndarray, list]:
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
    date_max = 1000
    df.loc[df['date_sympt'] != 1000, 'date_sympt_norm'] = df.loc[df['date_sympt'] != 1000, 'date_sympt'] / date_max
    df['end_followup_norm'] = df['end_followup'] / date_max
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


def plot_attention_scores(attention_scores,
                          head_idx: Optional[int] = None,
                          group_idx: Optional[int] = None,
                          normalize: bool = True):
    """
    Plots a heatmap of attention scores for a specific batch and head group.

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
    scores = tf.reduce_mean(scores, axis=0)  # average over batches
    if head_idx is None:
        scores = tf.reduce_mean(scores, axis=0).numpy()  # Shape: (9, 9)
    else:
        scores = scores[head_idx, :, :].numpy()  # Shape: (9, 9)

    # Normalize the scores
    if normalize:
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

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
    plt.show()

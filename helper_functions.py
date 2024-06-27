from typing import Union

import numpy as np
import pandas as pd
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
    df.loc[df['date_sympt'] == 1000, 'date_sympt_norm'] = 1  # when sorting it stays at the end
    df['end_followup_norm'] = df['end_followup'] / date_max

    # todo: what happens if infection date is after end of follow-up?

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

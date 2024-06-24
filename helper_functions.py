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


def normalize_household_data(df: pd.DataFrame,
                             minimal_length: int,
                             return_list: bool,
                             seed: int = 0) -> Union[np.ndarray, list]:
    """
    Normalizes the household data and returns it as a numpy array or list.
    Patients in a household are order by event date.

    Parameters
    ----------
    df : pd.DataFrame - the household data
    minimal_length : int - the minimal length of the household data: in long format, this is the total sequence length,
                    in list format, this is the minimal number of people in a household. In both cases, we use zero
                    padding at the beginning of the sequence to reach this length.
    return_list : bool - if True, returns a list of households each of size (time x features)
                    (or numpy array if minimal_length > 0), otherwise a numpy array in the format
                    (features x time) where households get an id as feature
    seed : int - random seed for shuffling the households
    """
    df = df.copy()
    all_households = []

    # date_sympt = 1000 is not infected
    date_sympt_max = np.max(df.loc[df['date_sympt'] != 1000, 'date_sympt'])
    date_end_followup_max = np.max(df['end_followup'])
    date_max = max(date_sympt_max, date_end_followup_max) + 1  # add 1 to avoid confusion with no infection date
    df.loc[df['date_sympt'] != 1000, 'date_sympt_norm'] = df.loc[df['date_sympt'] != 1000, 'date_sympt'] / date_max
    df.loc[df['date_sympt'] == 1000, 'date_sympt_norm'] = 1  # when sorting it stays at the end
    df['end_followup_norm'] = df['end_followup'] / date_max

    df['infect_status_norm'] = df['infect_status'] - 1
    df['age_norm'] = df['age'] - 1

    unique_households = df['id_hh'].unique()

    np.random.seed(seed)
    np.random.shuffle(unique_households)

    for i, id_hh in enumerate(unique_households):
        df_hh = df[df['id_hh'] == id_hh]
        if return_list:
            # household as a list, so no need to add an id
            household = np.stack((
                df_hh['date_sympt_norm'].values,
                df_hh['infect_status_norm'].values,
                df_hh['age_norm'].values,
                df_hh['protected'].values
            ))
            household = np.concatenate((
                household,
                [[df_hh['end_followup_norm'].values[0]],
                 [0], [0], [0]]
            ), axis=1)
        else:
            # also append an id to the household
            household = np.stack((
                df_hh['date_sympt_norm'].values,
                np.ones_like(df_hh['date_sympt_norm'].values) * i / (len(df['id_hh'].unique()) - 1),
                df_hh['infect_status_norm'].values,
                df_hh['age_norm'].values,
                df_hh['protected'].values
            ))
            household = np.concatenate((
                household,
                [[df_hh['end_followup_norm'].values[0]],
                 [i / (len(df['id_hh'].unique()) - 1)],
                 [0], [0], [0]]
            ), axis=1)

        # there is no specific order in the household, so order by date of symptoms
        order = np.argsort(household[0])
        household = household[:, order]

        if return_list and household.shape[1] < minimal_length:
            # pad sequence length with zeros, each household gets padded individually
            household = np.concatenate([np.zeros((household.shape[0],
                                                  minimal_length - household.shape[1])),
                                        household], axis=1)
        all_households.append(household)

    if return_list:
        if minimal_length > 0:
            return np.stack([h.T for h in all_households])
        return all_households

    all_households = np.concatenate(all_households, axis=1)
    if all_households.shape[1] < minimal_length:
        # pad sequence length with zeros, only total length is considered
        all_households = np.concatenate([np.zeros((all_households.shape[0],
                                                   minimal_length - all_households.shape[1])),
                                         all_households,
                                         ], axis=1)
    return all_households

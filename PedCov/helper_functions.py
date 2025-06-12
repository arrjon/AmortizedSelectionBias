from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
import pandas as pd


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    # Check if the list is empty
    if not list_of_dicts:
        return {}

    # Initialize the dictionary of arrays
    dict_of_arrays = {key: [] for key in list_of_dicts[0]}

    # Populate the dictionary of arrays
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            dict_of_arrays[key].append(value)
    for k in dict_of_arrays:
        dict_of_arrays[k] = np.array(dict_of_arrays[k])
    return dict_of_arrays


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

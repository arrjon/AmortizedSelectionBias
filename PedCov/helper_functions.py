from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats


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
    DATE_MAX: float = 200.  # maximum date in the dataset
    AGE_MAX: float = 100.
    USE_ONE_HOT: bool = True
    NOT_INFECTED_DATE: float = 1000.  # special value indicating not infected until end of follow up


# One-hot encoding dictionaries
ENCODING_DICT = {
    'infect_status': {
        InfectionStatus.NOT_INFECTED: [1, 0, 0],
        InfectionStatus.INFECTED_SYMPTOMATIC: [0, 1, 0],
        InfectionStatus.INFECTED_ASYMPTOMATIC: [0, 0, 1],
    },
    'age': {
        AgeGroup.INFANT: [0],
        AgeGroup.CHILD: [1],
        AgeGroup.OLDER: [2],
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
    required_columns = {'id_hh', 'date_sympt', 'infect_status', 'age_exact',
                        'protected', 'hh_size', 'end_followup', 'first_test_pos', 'last_test_neg'}
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
        encoded = []
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
    nan_mask = df['date_sympt'] != config.NOT_INFECTED_DATE
    df.loc[nan_mask, 'date_sympt_norm'] = df.loc[nan_mask, 'date_sympt'] / config.DATE_MAX
    df.loc[~nan_mask, 'date_sympt_norm'] = config.NOT_INFECTED_DATE  # later replaced with -1

    nan_mask = df['first_test_pos'] != config.NOT_INFECTED_DATE
    df.loc[nan_mask, 'first_test_pos_norm'] = df.loc[nan_mask, 'first_test_pos'] / config.DATE_MAX
    df.loc[~nan_mask, 'first_test_pos_norm'] = config.NOT_INFECTED_DATE  # later replaced with -1

    nan_mask = df['last_test_neg'] != -config.NOT_INFECTED_DATE  # negative value
    df.loc[nan_mask, 'last_test_neg_norm'] = df.loc[nan_mask, 'last_test_neg'] / config.DATE_MAX
    df.loc[~nan_mask, 'last_test_neg_norm'] = config.NOT_INFECTED_DATE  # later replaced with -1

    # Normalize end followup dates
    df['end_followup_norm'] = df['end_followup'] / config.DATE_MAX

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

        # Construct household array without follow-up
        household = np.concatenate((
            df_hh['date_sympt_norm'].values[:, np.newaxis],
            df_hh['last_test_neg_norm'].values[:, np.newaxis],
            df_hh['first_test_pos_norm'].values[:, np.newaxis],
            encoded_household,  # infection status, age group, protection status
            df_hh['age_exact_norm'].values[:, np.newaxis],
            df_hh['hh_size'].values[:, np.newaxis],  # household size
            df_hh['end_followup_norm'].values[:, np.newaxis],
        ), axis=1)

        # Sort only the main PedCov by date (excluding not infected)
        order = np.argsort(household[:, 0])
        household = household[order]

        # transpose to have shape (n_features, n_members)
        household = household.T

    else:
        # Construct main household PedCov
        household = np.stack((
            df_hh['date_sympt_norm'].values,
            df_hh['last_test_neg_norm'].values,
            df_hh['first_test_pos_norm'].values,
            df_hh['infect_status'].values,
            df_hh['age'].values,  # age_group
            df_hh['age_exact_norm'].values,
            df_hh['protected'].values,
            df_hh['hh_size'].values,
            df_hh['end_followup_norm'].values
        ))

        # Sort main PedCov by date
        order = np.argsort(household[0])
        household = household[:, order]

    # replace 1000 with -1
    household[household == config.NOT_INFECTED_DATE] = -1

    # Pad if necessary
    if household.shape[1] < minimal_length:
        padding = np.zeros((household.shape[0], minimal_length - household.shape[1]))
        household = np.concatenate([padding, household], axis=1)
    return household


def normalize_household_data(
        df: pd.DataFrame,
        minimal_length: int = 8,
        n_households: int = 128,
        config: ProcessingConfig = ProcessingConfig()
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Normalizes household PedCov and returns it as a numpy array or list.

    Args:
        df: Input DataFrame containing household PedCov
        minimal_length: Minimum sequence length required
        n_households: Number of households to process (default is 128). Will apply padding if less households are present.
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

        # Pad if necessary
        if len(all_households) < n_households:
            padding = np.zeros((minimal_length, all_households[0].shape[1]))
            all_households.extend([padding] * (n_households - len(all_households)))

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


# plot delay distribution
def plot_delay_distribution(delayDist):
    x = np.arange(0, 20, 0.1)
    y_symp = stats.gamma.pdf(x+delayDist[2], a=delayDist[0], scale=1/delayDist[1])
    y_asymp = stats.gamma.pdf(x+delayDist[5], a=delayDist[3], scale=1/delayDist[4])
    plt.plot(x, y_symp, label='Symptomatic')
    plt.plot(x, y_asymp, label='Asymptomatic')
    plt.xlabel('Days')
    plt.ylabel('Probability Density')
    plt.title('Delay Distribution')
    plt.legend()
    plt.show()

# plot incubation distribution
def plot_incubation_distribution(shape, scale, shape_asym, scale_asym):
    x = np.arange(0, 20, 0.1)
    y = stats.gamma.pdf(x, a=shape, scale=scale)
    plt.plot(x, y, label='Incubation Distribution Symptomatic')
    y = np.ones_like(x) / (7 - 2)  # uniform distribution for asymptomatic
    y[x < 2] = 0
    y[x > 7] = 0
    plt.plot(x, y, label='Incubation Distribution Asymptomatic (old)', linestyle='--')
    y = stats.gamma.pdf(x, a=shape_asym, scale=scale_asym)
    plt.plot(x, y, label='Incubation Distribution Asymptomatic')
    plt.xlabel('Days')
    plt.ylabel('Probability Density')
    plt.title('Incubation Distribution')
    plt.legend()
    plt.show()

# plot generation time distribution
def plot_generation_time_distribution(shapeInf, scaleInf):
    x = np.arange(0, 20, 0.1)
    y = stats.gamma.pdf(x, a=shapeInf, scale=scaleInf)
    plt.plot(x, y, label='Generation Time Distribution')
    plt.xlabel('Lag (days)')
    plt.ylabel('Probability Density')
    plt.title('Generation Time Distribution')
    plt.legend()
    plt.show()


########## this code is taken from pypesto
def calculate_ci(
    values: np.ndarray,
    ci_level: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate confidence/credibility levels using percentiles.

    Parameters
    ----------
    values:
        The values used to calculate percentiles.
    ci_level:
        Lower tail probability.
    kwargs:
        Additional keyword arguments are passed to the `numpy.percentile` call.

    Returns
    -------
    lb, ub:
        Bounds of the confidence/credibility interval.
    """
    # Percentile values corresponding to the CI level
    percentiles = 100 * np.array([(1 - ci_level) / 2, 1 - (1 - ci_level) / 2])
    # Upper and lower bounds
    lb, ub = np.percentile(values, percentiles, **kwargs)
    return lb, ub


def sampling_parameter_cis(
    posterior_samples: np.ndarray,
    param_names: list[str] = None,
    alpha: list[int] = None,
    step: float = 0.05,
    show_median: bool = True,
    title: str = None,
    size: tuple[float, float] = None,
    ax: matplotlib.axes.Axes = None,
) -> matplotlib.axes.Axes:
    """
    Plot MCMC-based parameter credibility intervals.

    Parameters
    ----------
    posterior_samples: ndarray
        MCMC samples of the posterior distribution, shape (n_samples, n_parameters).
    param_names: list of str
        Names of the parameters, defaults to None.
    alpha:
        List of lower tail probabilities, defaults to 95% interval.
    step:
        Height of boxes for projectile plot, defaults to 0.05.
    show_median:
        Plot the median of the MCMC chain. Default: True.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """
    if alpha is None:
        alpha = [95]

    # automatically sort values in decreasing order
    alpha_sorted = sorted(alpha, reverse=True)
    # define colormap
    evenly_spaced_interval = np.linspace(0, 1, len(alpha_sorted))
    colors = [plt.cm.tab20c_r(x) for x in evenly_spaced_interval]
    # number of sampled parameters
    n_pars = posterior_samples.shape[-1]

    # set axes and figure
    if ax is None:
        _, ax = plt.subplots(figsize=size, layout='constrained')

    # loop over parameters
    for npar in range(n_pars):
        # initialize height of boxes
        _step = step
        # loop over confidence levels
        for n, level in enumerate(alpha_sorted):
            # extract percentile-based confidence intervals
            lb, ub = calculate_ci(posterior_samples, ci_level=level / 100, axis=0)

            # assemble boxes for projectile plot
            x1 = [lb[npar], ub[npar]]
            y1 = [npar + _step, npar + _step]
            y2 = [npar - _step, npar - _step]
            # Plot boxes
            ax.fill(
                np.append(x1, x1[::-1]),
                np.append(y1, y2[::-1]),
                color=colors[n],
                label=str(level) + "% CI",
            )

            if show_median:
                if n == len(alpha_sorted) - 1:
                    _median = np.median(posterior_samples[:, npar])
                    ax.plot(
                        [_median, _median],
                        [npar - _step, npar + _step],
                        "k-",
                        label="Median",
                    )

            # increment height of boxes
            _step += step

    ax.set_yticks(range(n_pars))
    if param_names is not None:
        ax.set_yticklabels(param_names)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Parameter name")

    if title:
        ax.set_title(title)

    # handle legend
    plt.gca().invert_yaxis()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1))
    return ax


def sampling_parameter_cis_comparison(
    results: dict[str, dict],
    methods: dict[str, str],
    variant: str,
    param_dict: dict[str, str] = None,
    alpha: list[int] = None,
    step: float = 0.05,
    show_median: bool = True,
    title: str = None,
    size: tuple[float, float] = None,
    ax: matplotlib.axes.Axes = None,
    show_legend: bool = False
) -> matplotlib.axes.Axes:
    """
    Plot MCMC-based parameter credibility intervals for multiple methods,
    using colored boxes per method.
    """
    if alpha is None:
        alpha = [95]
    alpha_sorted = sorted(alpha, reverse=True)
    n_levels = len(alpha_sorted)

    # pick a distinct color for each method
    cmap = plt.get_cmap("tab10")
    method_colors = [cmap(i) for i in range(len(methods))]

    # number of parameters
    sample0 = results[list(methods.keys())[0]]
    n_pars = len(sample0)

    # vertical offsets so that each method is centered on its own y
    height_per_method = step * (n_levels * 2 + 1)
    method_offsets = [
        (i - (len(methods)-1)/2) * height_per_method
        for i in range(len(methods))
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=size, layout='constrained')

    # draw each method
    for vi, method in enumerate(methods.keys()):
        samples = np.stack([
            results[method][p][0, :, 0]
            for p in param_dict.keys()
        ], axis=-1)

        for npar in range(n_pars):
            base_y = npar + method_offsets[vi]
            _step = step

            for lvl_i, level in enumerate(alpha_sorted):
                lb, ub = calculate_ci(samples, ci_level=level/100, axis=0)

                xs = [lb[npar], ub[npar], ub[npar], lb[npar]]
                ys = [
                    base_y - _step,
                    base_y - _step,
                    base_y + _step,
                    base_y + _step
                ]
                # fill with variant color, alpha lighter for smaller CIs
                fill_alpha = 0.3 + 0.5 * (lvl_i / (n_levels-1) if n_levels>1 else 1)
                ax.fill(
                    xs, ys,
                    facecolor=method_colors[vi],
                    edgecolor=method_colors[vi],
                    alpha=fill_alpha,
                )
                _step += step

            # median line
            if show_median:
                med = np.median(samples[:, npar])
                ax.plot(
                    [med, med],
                    [base_y - step, base_y + step],
                    color='black'
                )

    # styling
    ax.axvline(1, color='grey', linestyle='--')
    ax.set_yticks(range(n_pars))
    if param_dict is not None:
        ax.set_yticklabels(param_dict.values(), fontsize=12)
    ax.set_xlabel(f"{variant.title()} parameter value")
    ax.set_ylabel("Parameters")
    if title:
        ax.set_title(title)
    ax.invert_yaxis()

    # legend: grey patches for CI levels + colored lines for methods
    ci_patches = [
        Patch(facecolor=str(0.8 - i*0.2), edgecolor='none', label=f"{level}% CI")
        for i, level in enumerate(alpha_sorted)
    ]
    variant_lines = [
        Patch(color=method_colors[i], label=name)
        for i, name in enumerate(methods.values())
    ]
    if show_legend:
        ax.legend(
            handles = ci_patches + variant_lines,
            bbox_to_anchor=(1,1),
            frameon=False,
            fontsize=12
        )
    return ax


def find_best_model_by_rank_sum(list1, list2):
    """
    Find best model using sum of rankings approach
    list1, list2: lists of scores where index represents the model
    Returns: index of the best model
    """
    n = len(list1)

    # Create rankings (rank 0 = best, rank n-1 = worst)
    # Sort indices by their values in descending order
    rank1 = [0] * n
    rank2 = [0] * n

    # Get sorted indices for list1 (best to worst)
    sorted_indices1 = sorted(range(n), key=lambda i: list1[i])
    for rank, idx in enumerate(sorted_indices1):
        rank1[idx] = rank

    # Get sorted indices for list2 (best to worst)
    sorted_indices2 = sorted(range(n), key=lambda i: list2[i])
    for rank, idx in enumerate(sorted_indices2):
        rank2[idx] = rank

    # Sum rankings for each model
    combined_ranks = [rank1[i] + rank2[i] for i in range(n)]

    # Best model has lowest combined rank
    return combined_ranks.index(min(combined_ranks))


def find_best_model_by_normalized_avg(list1, list2):
    """
    Find best model using normalized score average
    list1, list2: lists of scores where index represents the model
    Returns: index of the best model with the lowest average normalized score
    """
    # Normalize scores to 0-1 range
    def normalize(scores):
        min_val, max_val = min(scores), max(scores)
        if max_val == min_val:  # Handle edge case where all scores are equal
            return [0.5] * len(scores)
        return [(score - min_val) / (max_val - min_val) for score in scores]

    norm1 = normalize(list1)
    norm2 = normalize(list2)

    # Calculate average normalized scores
    avg_scores = [(norm1[i] + norm2[i]) / 2 for i in range(len(list1))]

    # Return index of highest average score
    return avg_scores.index(min(avg_scores))

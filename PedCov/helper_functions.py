from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Union
from collections import Counter

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
    show_legend: bool = False,
    colors: list[str] = None,
):
    """
    Plot MCMC-based parameter credibility intervals for multiple methods,
    using colored boxes per method.
    """
    if alpha is None:
        alpha = [95]
    alpha_sorted = sorted(alpha, reverse=True)
    n_levels = len(alpha_sorted)

    if colors is None:
        # pick a distinct color for each method
        cmap = plt.get_cmap("tab10")
        method_colors = [cmap(i) for i in range(len(methods))]
    else:
        method_colors = colors
        if len(method_colors) < len(methods):
            raise ValueError("Not enough colors provided for the number of methods.")

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
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_yticks(range(n_pars))
    if param_dict is not None:
        ax.set_yticklabels(param_dict.values(), fontsize=10)
    ax.set_xlabel(f"{variant.title()} parameter value", fontsize=10)
    ax.set_ylabel("Parameters", fontsize=10)
    if title:
        ax.set_title(title)
    ax.invert_yaxis()

    # legend: grey patches for CI levels + colored lines for methods
    ci_patches = [
        Patch(facecolor=str(0.8 - i*0.2), edgecolor='none', label=f"{level}% CI")
        for i, level in enumerate(alpha_sorted)
    ][::-1]
    variant_lines = [
        Patch(color=method_colors[i], label=name)
        for i, name in enumerate(methods.values())
    ][::-1]
    handles = ci_patches + variant_lines
    if show_legend:
        ax.legend(
            handles = handles,
            bbox_to_anchor=(0.5, 1),
            frameon=True,
            fontsize=10,
            facecolor='white',
            edgecolor='white',
        )
    return ax, handles


def count_households_first_pos(
        real_data_results,
        sim_key: str = "sim_data",
        feature_idx_first_pos: int = 2,
        feature_idx_age_group: int = 6,
) -> dict:
    """
    Counts (non-padded) households per variant and which age group had the
    earliest positive test in each household.

    Expected household tensor layout per household row (member):
      0  date_sympt_norm
      1  last_test_neg_norm
      2  first_test_pos_norm
      3  infect_status_onehot[0], not infected status
      4  infect_status_onehot[1], symptomatic status
      5  infect_status_onehot[2], asymptomatic status
      6  age_group   (0=INFANT, 1=CHILD, 2=OLDER)
      7  protected
      8  age_exact_norm
      9  hh_size
      10 end_followup_norm

    sim_data can be:
      - (n_households, n_members, n_features)
      - (n_batches, n_households, n_members, n_features)
    """

    out = {}

    def _is_padding_member(row: np.ndarray) -> bool:
        # Padding rows are all zeros
        if np.allclose(row, 0.0):
            return True
        return False

    def _is_padding_household(hh: np.ndarray) -> bool:
        # Household is padded if all members are padding rows
        return all(_is_padding_member(member) for member in hh)


    data_dict = real_data_results.copy()
    flat_data = False
    if "variant" in data_dict:
        flat_data = True

    if flat_data:
        data_dict = {}
        if 'alpha' in real_data_results['variant']:
            data_dict['alpha'] = {'sim_data': real_data_results['sim_data'][real_data_results['variant'] == 'alpha']}
        if 'omicron' in real_data_results['variant']:
            data_dict['omicron'] = {'sim_data': real_data_results['sim_data'][real_data_results['variant'] == 'omicron']}

    for variant, data in data_dict.items():
        if sim_key not in data:
            raise KeyError(f"Variant '{variant}' missing key '{sim_key}'")

        sim_data = np.asarray(data[sim_key])

        if sim_data.ndim == 3:
            all_households = sim_data[None]  # (1, n_households, n_members, n_features)
        elif sim_data.ndim == 4:
            all_households = sim_data  # (n_batches, n_households, n_members, n_features)
        else:
            raise ValueError(f"sim_data for variant '{variant}' has unsupported ndim={sim_data.ndim}")

        out[variant] = {
            "n_households": [],
            "first_positive_age_group_counts": [],
        }
        for households in all_households:
            counts = Counter()
            n_valid_households = 0

            for hh in households:
                if _is_padding_household(hh):
                    continue  # exclude fully padded households

                n_valid_households += 1

                first_pos = np.inf
                first_age_group = None
                first_member = None

                for member in hh:
                    if _is_padding_member(member):
                        continue

                    t = member[feature_idx_first_pos]

                    if t < first_pos:
                        first_pos = t
                        first_age_group = int(member[feature_idx_age_group])
                        first_member = member
                    elif t == first_pos:
                        # tie-breaking: if two members have the same first positive time, we check for symptomatic status
                        # if one is symptomatic and the other is asymptomatic, we prioritize the symptomatic one as first positive
                        if member[4] == 1 and first_member[4] == 0:
                            first_age_group = int(member[feature_idx_age_group])
                            first_member = member
                        elif member[4] == 0 and first_member[4] == 1:
                            # keep the existing first positive as it is symptomatic
                            pass
                        else:
                            # take the younger one
                            if member[8] < first_member[8]:  # compare age_exact_norm
                                first_age_group = int(member[feature_idx_age_group])
                                first_member = member

                if first_age_group is None:
                    counts['no_pos'] += 1
                else:
                    counts[first_age_group] += 1

            # ensure keys exist
            age_group_counts = {0: 0, 1: 0, 2: 0}
            for k, v in counts.items():
                age_group_counts[k] = int(v)

            out[variant]["n_households"].append(n_valid_households)
            out[variant]["first_positive_age_group_counts"].append(age_group_counts)

            assert np.sum(list(age_group_counts.values())) == n_valid_households, f"Variant '{variant}': sum of age group counts {age_group_counts.values().sum()} does not match number of valid households {n_valid_households}"

    for variant, data in data_dict.items():
        out[variant]['first_positive_age_group_counts'] = list_of_dicts_to_dict_of_lists(out[variant]['first_positive_age_group_counts'])
        assert all(np.array(out[variant]['n_households']) == out[variant]['n_households'][0]), f"Variant '{variant}': number of households varies across batches: {out[variant]['n_households']}"
        out[variant]['n_households'] = out[variant]['n_households'][0]
    return out


def plot_first_positive_age_group_counts(
        list_sim: list[dict], colors: list[str], labels: list[str],
        real_data: dict = None, save_path = None
):

    first_positive_dicts = [count_households_first_pos(data) for data in list_sim]

    if real_data is not None:
        real_data = count_households_first_pos(real_data)

    fig, ax = plt.subplots(ncols=1, nrows=len(first_positive_dicts[0]), figsize=(5, 4),
                           sharey=True, sharex=True, layout='constrained')
    ax = ax.flatten()
    width = 0.2
    offsets = np.linspace(-width, width, len(first_positive_dicts))
    age_groups = ['Infant', 'Child', 'Adult']
    for i, first_positive_dict in enumerate(first_positive_dicts):
        for a, (variant, data) in zip(ax, first_positive_dict.items()):
            counts = data['first_positive_age_group_counts']
            parts = a.violinplot(
                np.array([counts[0], counts[1], counts[2]]).T / data['n_households'],
                positions=np.arange(len(age_groups)) + offsets[i],
                widths=0.15,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            a.set_ylabel(f"{variant}".title(), fontsize=12)
            # Label each set of violin bodies for legend
            for body in parts['bodies']:
                body.set_color(colors[i])
                body.set_alpha(1)
            parts['cmedians'].set_color(colors[i])
            if variant == 'alpha':
                for body in parts['bodies']:
                    body.set_label(labels[i])
                    break

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.grid(axis='y')
        a.set_ylim(0, None)
        a.set_xticks(np.arange(len(age_groups)))
        a.set_xticklabels(age_groups, fontsize=12)

    if real_data is not None:
        for a, (variant, _) in zip(ax, first_positive_dicts[0].items()):
            real_counts = real_data[variant]['first_positive_age_group_counts']
            a.scatter(
                np.arange(len(age_groups)),
                np.array([real_counts[0], real_counts[1], real_counts[2]]) / real_data[variant]['n_households'],
                color='black',
                marker='x',
                s=75,
                label='Real Data' if variant == 'alpha' else None,
                zorder=3
            )

    fig.supylabel("Fraction of age group with first positive", fontsize=12)

    handles, labels = ax[0].get_legend_handles_labels()
    # move the real data handle to the second row
    handles = [handles[0], handles[-1]] + handles[1:-1]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncols=3, frameon=False, fontsize=10)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

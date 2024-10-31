import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


STATUS_NAME = ['not_infected', 'infected_symptomatic', 'infected_asymptomatic']
COLOR_AGE = {
    '<6 years': '#b2df8a',
    '6-11 years': '#d95f02',
    '>11 years': '#1f78b4'
}
COLOR_STATUS = {
    'infected_symptomatic': '#1f78b4',
    'infected_asymptomatic': '#b2df8a'
}


# this code is adapted from the pypesto package
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


# this code is adapted from the pypesto package
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
    result:
        The pyPESTO result object with filled sample result.
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
        _, ax = plt.subplots(figsize=size, tight_layout=True)

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


def plot_household_statistic(real_data_stats, sim_data_stats, results_folder):
    # plot the distribution of number of infected individuals
    for variant in ['alpha', 'omicron']:
        print(f"Variant: {variant}")
        fig, ax = plt.subplots(2, len(real_data_stats[variant]['household_sizes']),
                               sharex=True, sharey='col', tight_layout=True, figsize=(12, 8))
        for i, household_size in enumerate(real_data_stats[variant]['household_sizes']):
            bottom = {0: 0, 1: 0, 2: 0}
            for age_group in real_data_stats[variant]['infection_counts'].index:
                infected_dict = real_data_stats[variant]['infection_counts'].loc[age_group, household_size]
                if not isinstance(infected_dict, dict):
                    continue

                ax[0, i].bar(infected_dict.keys(), infected_dict.values(),
                             bottom=[bottom[k] for k in infected_dict.keys()],
                             alpha=0.5, label=age_group, color=COLOR_AGE[age_group])
                bottom.update({k: bottom[k] + infected_dict[k] for k in infected_dict.keys()})
            ax[0, i].set_title(f'Household Size {household_size}')

        for i, household_size in enumerate(
                real_data_stats[variant]['household_sizes']):  # household size same as in real data
            bottom = {0: 0, 1: 0, 2: 0}
            for age_group in sim_data_stats[variant]['infection_counts'].index:
                if household_size not in sim_data_stats[variant]['infection_counts'].columns:
                    continue
                infected_dict = sim_data_stats[variant]['infection_counts'].loc[age_group, household_size]
                if not isinstance(infected_dict, dict):
                    continue

                ax[1, i].bar(infected_dict.keys(), infected_dict.values(),
                             bottom=[bottom[k] for k in infected_dict.keys()],
                             alpha=0.5, label=age_group, color=COLOR_AGE[age_group])
                bottom.update({k: bottom[k] + infected_dict[k] for k in infected_dict.keys()})
            ax[1, i].set_xticks(ticks=[0, 1, 2], labels=STATUS_NAME, rotation=45)

        ax[0, 0].set_ylabel('Frequency (Real Data)')
        ax[1, 0].set_ylabel('Frequency (Sim Data)')
        handles = [Patch(facecolor=color, label=age_group, alpha=0.5) for age_group, color in COLOR_AGE.items()]
        fig.legend(handles, COLOR_AGE.keys(),
                   loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
        plt.savefig(f'{results_folder}/real data/number_infections_{variant}.png', bbox_inches='tight')
        plt.show()

    # plot the distribution of infection times
    for variant in ['alpha', 'omicron']:
        for cofactor_type in ['time_distribution', 'time_distribution_age']:
            colors = COLOR_AGE if cofactor_type == 'time_distribution_age' else COLOR_STATUS
            print(f"Variant: {variant}")
            fig, ax = plt.subplots(2, len(real_data_stats[variant]['household_sizes']),
                                   sharex='col', sharey='col', tight_layout=True, figsize=(12, 8))
            for i, household_size in enumerate(real_data_stats[variant]['household_sizes']):
                for status in real_data_stats[variant][cofactor_type].index:
                    if status == 'not_infected':
                        continue
                    infection_times = np.array(real_data_stats[variant][cofactor_type].loc[status, household_size])
                    if np.isnan(infection_times).all():
                        continue
                    ax[0, i].hist(infection_times * 1000, bins=10, alpha=0.5, label=status, color=colors[status])
                ax[0, i].set_title(f'Household Size {household_size}')

            for i, household_size in enumerate(
                    real_data_stats[variant]['household_sizes']):  # household size same as in real data
                for status in sim_data_stats[variant][cofactor_type].index:
                    if status == 'not_infected' or household_size not in sim_data_stats[variant][cofactor_type].columns:
                        continue
                    infection_times = np.array(sim_data_stats[variant][cofactor_type].loc[status, household_size])
                    if np.isnan(infection_times).all():
                        continue
                    ax[1, i].hist(infection_times * 1000, bins=10, alpha=0.5, label=status, color=colors[status])
                ax[1, i].set_xlabel('Infection Date')

            ax[0, 0].set_ylabel('Frequency (Real Data)')
            ax[1, 0].set_ylabel('Frequency (Sim Data)')
            handles = [Patch(facecolor=color, label=status, alpha=0.5) for status, color in colors.items()]
            fig.legend(handles, colors.keys(),
                       loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
            plt.savefig(f'{results_folder}/real data/infection_time_points_{variant}_{cofactor_type}.png',
                        bbox_inches='tight')
            plt.show()

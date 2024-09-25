# this code is adapted from the pypesto package

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
                        label="MCMC median",
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
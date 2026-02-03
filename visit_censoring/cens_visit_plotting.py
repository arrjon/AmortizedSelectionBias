import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid

colors = ['#4B2E83', '#D64A62', '#E68600', '#1B8A8F']
weibull_param_map = {
    '01': ('a01', 'shape01'),
    '02': ('a02', 'shape02'),
    '12': ('a12', 'shape12'),
}
epochs = ["epoch1", "epoch2", "epoch3", "epoch4"]
params_beta = ['beta01_age', 'beta02_age', 'beta12_age', 'beta01_sex', 'beta02_sex', 'beta12_sex']
params_a = ['a01', 'a02', 'a12']
params_shape = ['shape01', 'shape02', 'shape12']


def cumulative_trapz(t, h):
    result = cumulative_trapezoid(h, x=t, initial=0.0)
    return result

def cumulative_trapz_samples(t, h_samples):
    """
    Cumulative integral for an array of curves.
    h_samples: shape (n_samples, n_time)
    """
    # cumulative_trapezoid works along axis, so we integrate along axis=1 (time axis)
    result = cumulative_trapezoid(h_samples, x=t, axis=1, initial=0.0)
    return result

def weibull_hazard(t, a, shape):
    # h(t) = a * shape * t^(shape - 1)
    t = np.asarray(t, dtype=float)
    return a * shape * np.power(t, shape - 1)


def plot_params(
    baseline,
    prior_samples,
    prior_summary,
    posterior_samples,
    posterior_summary,
    network_name,
    save_path=None,
):
    transitions = ['01', '02', '12']
    beta_params = ['age', 'sex']
    hazard_params = ['a01', 'a02', 'a12']

    n_epochs = len(epochs)
    nrows = 2 + len(hazard_params)  # beta_age, beta_sex, hazards
    ncols = n_epochs

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3 * ncols, 2 * nrows),
        layout='constrained',
        sharey='row',
    )

    # ------------------------------------------------------------
    # 1) Beta rows (age, sex)
    # ------------------------------------------------------------
    for row_idx, beta in enumerate(beta_params):
        for col_idx, e in enumerate(epochs):
            this_ax = ax[row_idx, col_idx]

            vals_naive = [baseline[e][f'beta{t}_{beta}']['naive_cox'] for t in transitions]
            vals_weib = [baseline[e][f'beta{t}_{beta}']['weibull'] for t in transitions]
            vals_spl = [baseline[e][f'beta{t}_{beta}']['splines'] for t in transitions]

            x = np.arange(len(transitions))

            this_ax.plot(x, vals_naive, marker='o', color=colors[0], linestyle='dotted', zorder=3,
                         label='Naive Cox' if (row_idx == 0 and col_idx == 0) else None)
            this_ax.plot(x, vals_weib, marker='o', color=colors[1], linestyle='dotted', zorder=4, alpha=0.75,
                         label='Weibull IDM' if (row_idx == 0 and col_idx == 0) else None)
            this_ax.plot(x, vals_spl, marker='o', color=colors[2], linestyle='dotted', zorder=4, alpha=0.75,
                         label='Splines IDM' if (row_idx == 0 and col_idx == 0) else None)

            # prior band (constant over transitions)
            q = [prior_summary[f'beta{t}_{beta}'] for t in transitions]
            if len(q) > 0:
                this_ax.fill_between(
                    x,
                    [q[i]['low'] for i in range(len(transitions))],
                    [q[i]['high'] for i in range(len(transitions))],
                    color='black',
                    alpha=0.15,
                    label='Prior 95% interval' if (row_idx == 0 and col_idx == 0) else None,
                    zorder=1,
                )

            # posterior band (epoch-specific)
            q = [posterior_summary[f'beta{t}_{beta}'] for t in transitions]
            if len(q) > 0:
                this_ax.fill_between(
                    x,
                    [q[i]['low'][col_idx] for i in range(len(transitions))],
                    [q[i]['high'][col_idx] for i in range(len(transitions))],
                    color=colors[3],
                    alpha=0.3,
                    label='Posterior 95% interval' if (row_idx == 0 and col_idx == 0) else None,
                    zorder=2,
                )
                this_ax.plot(
                    x,
                    [q[i]['median'][col_idx] for i in range(len(transitions))],
                    linestyle='--',
                    color=colors[3],
                    label='Posterior median' if (row_idx == 0 and col_idx == 0) else None,
                    zorder=5,
                )

            this_ax.set_xticks(x)
            this_ax.set_xticklabels(transitions)
            if col_idx == 0:
                this_ax.set_ylabel(beta.replace('_', ' '))
            if row_idx == 0:
                this_ax.set_title(f'Epoch {e[-1]}')
            else:
                this_ax.set_xlabel('Transition')

            # remove top and right spines
            this_ax.spines['top'].set_visible(False)
            this_ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------
    # 2) Hazard rows (a01, a02, a12)
    # ------------------------------------------------------------
    for h_idx, a_name in enumerate(hazard_params):
        row = 2 + h_idx

        trans = a_name[-2:]  # '01', '02', '12'
        a_key, shape_key = weibull_param_map[trans]

        for col_idx, e in enumerate(epochs):
            this_ax = ax[row, col_idx]

            # Naive Cox reference
            this_ax.axhline(
                baseline[e][a_name]['naive_cox'],
                color=colors[0],
                label='Naive Cox' if (row == 2 and col_idx == 0) else None,
                zorder=3,
            )

            # IDM curves
            for b_i, b in enumerate(['idm_weib', 'idm_splines']):
                t_days = baseline[e][f'{b}_times']
                t_years = np.asarray(t_days) / 365.25
                haz = baseline[e][f'{b}_{trans}']

                this_ax.plot(
                    t_years,
                    haz,
                    color=colors[b_i + 1],
                    label=('Weibull IDM' if b == 'idm_weib' else 'Splines IDM')
                    if (row == 2 and col_idx == 0) else None,
                    zorder=4,  alpha=0.75,
                )

            # time grid for prior/posterior
            times_all = np.array(baseline[e]['idm_weib_times'])
            t_grid = np.linspace(1e-3, times_all.max(), 200)
            t_grid_years = t_grid / 365.25

            # prior hazard
            if a_key in prior_samples and shape_key in prior_samples:
                haz_samp = weibull_hazard(
                    t_grid[None, :],
                    prior_samples[a_key][:, None],
                    prior_samples[shape_key][:, None],
                )
                this_ax.fill_between(
                    t_grid_years,
                    np.quantile(haz_samp, 0.025, axis=0),
                    np.quantile(haz_samp, 0.975, axis=0),
                    color='black',
                    alpha=0.15,
                    label='Prior 95% interval' if (row == 2 and col_idx == 0) else None,
                    zorder=1,
                )

            # posterior hazard (epoch-specific)
            if a_key in posterior_samples and shape_key in posterior_samples:
                a_s = posterior_samples[a_key][col_idx].flatten()
                s_s = posterior_samples[shape_key][col_idx].flatten()

                haz_samp = weibull_hazard(
                    t_grid[None, :],
                    a_s[:, None],
                    s_s[:, None],
                )

                this_ax.fill_between(
                    t_grid_years,
                    np.quantile(haz_samp, 0.025, axis=0),
                    np.quantile(haz_samp, 0.975, axis=0),
                    color=colors[3],
                    alpha=0.3,
                    label='Posterior 95% interval' if (row == 2 and col_idx == 0) else None,
                    zorder=2,
                )
                this_ax.plot(
                    t_grid_years,
                    np.median(haz_samp, axis=0),
                    linestyle='--',
                    color=colors[3],
                    label='Posterior median' if (row == 2 and col_idx == 0) else None,
                    zorder=5,
                )

            this_ax.set_yscale('log')
            if col_idx == 0:
                this_ax.set_ylabel(a_name)
            if row == nrows - 1:
                this_ax.set_xlabel('Follow up years since entry in epoch')

            # remove top and right spines
            this_ax.spines['top'].set_visible(False)
            this_ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------
    # Global legend
    # ------------------------------------------------------------
    legend_handles = [
        Patch(facecolor=colors[0], label='Naive Cox'),
        Patch(facecolor=colors[1], alpha=0.75, label='Weibull IDM'),
        Patch(facecolor=colors[2],  alpha=0.75, label='Splines IDM'),
        Line2D([0], [0], color=colors[3], linestyle='--', label='Posterior median'),
        Patch(facecolor=colors[3], alpha=0.3, label='Posterior 95% interval'),
        Patch(facecolor='black', alpha=0.15, label='Prior 95% interval'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        ncols=3,
    )

    if save_path is not None:
        fig.savefig(save_path / f'{network_name}_params.png', bbox_inches='tight')
    plt.show()
    return


def plot_cumhaz(baseline, posterior_samples, df_real, network_name, per_person=100, adjust_cov=True, save_path=None):
    n_epochs = len(epochs)
    fig, axes = plt.subplots(1, n_epochs, figsize=(3 * n_epochs, 4), layout='constrained', sharey=True)

    # Handle case of single epoch
    if n_epochs == 1:
        axes = [axes]

    trans = '01'
    age_coef_name = f'beta{trans}_age'
    sex_coef_name = f'beta{trans}_sex'
    a01_param_name = f'a{trans}'

    # Weibull parameter names for transition 0→1
    a_key, shape_key = weibull_param_map[trans]

    # ----------------------------------------------------------------------
    # Time grid for prior/posterior cumulative hazards (epoch-local)
    # ----------------------------------------------------------------------
    epoch_lengths = {}
    for e in epochs:
        t_weib_e = np.asarray(baseline[e]['idm_weib_times'])
        epoch_lengths[e] = t_weib_e[-1] - t_weib_e[0]

    max_length = max(epoch_lengths.values())
    t_local = np.linspace(1e-3, max_length, 200)

    # check posterior availability
    has_posterior = (
            a_key in posterior_samples
            and shape_key in posterior_samples
            and age_coef_name in posterior_samples
            and sex_coef_name in posterior_samples
    )

    # ----------------------------------------------------------------------
    # Create one subplot per epoch
    # ----------------------------------------------------------------------
    for epoch_idx, e in enumerate(epochs):
        ax = axes[epoch_idx]

        # subset data for this epoch
        df_e = df_real[df_real['epoch'] == e]
        ages = df_e['age'].to_numpy()
        sexs = df_e['sex'].to_numpy()

        # ------------------------------------------------------------------
        # 1) Age/sex-adjusted point curves: Naive Cox, Weibull IDM, Splines IDM
        # ------------------------------------------------------------------
        # Naive Cox
        lp_naive = (
                baseline[e][age_coef_name]['naive_cox'] * ages +
                baseline[e][sex_coef_name]['naive_cox'] * sexs
        )
        w_bar_naive = np.exp(np.mean(lp_naive))

        # Weibull IDM
        lp_weib = (
                baseline[e][age_coef_name]['weibull'] * ages +
                baseline[e][sex_coef_name]['weibull'] * sexs
        )
        w_bar_weib = np.exp(np.mean(lp_weib))

        # Splines IDM
        lp_spl = (
                baseline[e][age_coef_name]['splines'] * ages +
                baseline[e][sex_coef_name]['splines'] * sexs
        )
        w_bar_spl = np.exp(np.mean(lp_spl))

        # Weibull IDM baseline hazard and cumulative hazard (epoch-local)
        t_weib = np.asarray(baseline[e]['idm_weib_times'])
        h_weib_base = np.asarray(baseline[e][f'idm_weib_{trans}'])
        if adjust_cov:
            h_weib_adj = h_weib_base * w_bar_weib
        else:
            h_weib_adj = h_weib_base
        H_weib_local = per_person * cumulative_trapz(t_weib, h_weib_adj)  # starts at 0

        # Splines IDM
        t_spl = np.asarray(baseline[e]['idm_splines_times'])
        h_spl_base = np.asarray(baseline[e][f'idm_splines_{trans}'])
        if adjust_cov:
            h_spl_adj = h_spl_base * w_bar_spl
        else:
            h_spl_adj = h_spl_base
        H_spl_local = per_person * cumulative_trapz(t_spl, h_spl_adj)

        # Naive Cox a01 (piecewise constant over epoch)
        h_naive_base = baseline[e][a01_param_name]['naive_cox']
        if adjust_cov:
            h_naive_adj = h_naive_base * w_bar_naive
        else:
            h_naive_adj = h_naive_base
        t_naive = t_weib
        H_naive_local = per_person * cumulative_trapz(
            t_naive,
            np.full_like(t_naive, h_naive_adj, dtype=float),
        )

        # Convert to years for plotting
        t_weib_years = (t_weib - t_weib[0]) / 365
        t_spl_years = (t_spl - t_spl[0]) / 365
        t_naive_years = (t_naive - t_naive[0]) / 365

        # Plot point estimates
        ax.plot(t_naive_years, H_naive_local, color=colors[0],
                label='Naive Cox' if epoch_idx == 0 else None)
        ax.plot(t_weib_years, H_weib_local, color=colors[1],
                label='Weibull IDM' if epoch_idx == 0 else None)
        ax.plot(t_spl_years, H_spl_local, color=colors[2],
                label='Splines IDM' if epoch_idx == 0 else None)

        # ------------------------------------------------------------------
        # 2) Posterior cumulative hazard band (age/sex adjusted, epoch-specific)
        # ------------------------------------------------------------------
        if has_posterior:
            a_samp_epoch = np.asarray(posterior_samples[a_key][epoch_idx]).flatten()
            shape_samp_epoch = np.asarray(posterior_samples[shape_key][epoch_idx]).flatten()
            beta_age_post_epoch = np.asarray(posterior_samples[age_coef_name][epoch_idx]).flatten()
            beta_sex_post_epoch = np.asarray(posterior_samples[sex_coef_name][epoch_idx]).flatten()

            n_samp_post = a_samp_epoch.shape[0]
            epoch_len = epoch_lengths[e]
            mask = t_local <= epoch_len
            t_grid_e = t_local[mask]

            h_post_samples = np.empty((n_samp_post, t_grid_e.size))
            for k in range(n_samp_post):
                lp_k = beta_age_post_epoch[k] * ages + beta_sex_post_epoch[k] * sexs
                w_bar_k = np.exp(np.mean(lp_k))

                h_k = weibull_hazard(t_grid_e, a_samp_epoch[k], shape_samp_epoch[k])
                if adjust_cov:
                    h_post_samples[k, :] = h_k * w_bar_k
                else:
                    h_post_samples[k, :] = h_k

            H_post_samples = per_person * cumulative_trapz_samples(t_grid_e, h_post_samples)
            med_e = np.median(H_post_samples, axis=0)
            low_e = np.quantile(H_post_samples, 0.025, axis=0)
            high_e = np.quantile(H_post_samples, 0.975, axis=0)

            t_grid_years = t_grid_e / 365
            ax.plot(
                t_grid_years,
                med_e,
                linestyle='--',
                color=colors[3],
                label='Posterior Median' if epoch_idx == 0 else None,
            )
            ax.fill_between(
                t_grid_years,
                low_e,
                high_e,
                alpha=0.15,
                color=colors[3],
                label='Posterior 95% CI' if epoch_idx == 0 else None,
            )
        ax.set_title(f'Epoch {e[-1]}')

        # Only show ylabel on leftmost subplot
        if epoch_idx == 0:
            if adjust_cov:
                ax.set_ylabel(r'Age and sex adjusted cumulative dementia hazard' + f'\nper {per_person} persons')
            else:
                ax.set_ylabel(f'Cumulative dementia hazard per {per_person} persons')

        # remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.supxlabel('Follow up years since entry in epoch')
    # Single legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))

    if save_path is not None:
        plt.savefig(save_path / f'{network_name}_{a01_param_name}_cumhaz_age_sex.png', bbox_inches='tight')
    plt.show()
    return

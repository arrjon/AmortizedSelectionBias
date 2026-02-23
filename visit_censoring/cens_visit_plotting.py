import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid
import plotly.graph_objects as go


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


def plot_params_error(
    posterior_samples_model,
    validation_data, validation_data_full,
    labels, param_names, param_names_pretty, colors_ordered, save_path=None
):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(5, 3.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
        layout="constrained",
    )

    base_positions = np.arange(len(param_names_pretty))
    width = 0.2
    offsets = [-width, 0, width]

    for i, ((model_name, samples), color) in enumerate(
            zip(posterior_samples_model.items(), colors_ordered)
    ):
        param_errors = []
        for p in param_names:
            ps = np.asarray(samples[p])
            if model_name == 'NPE (Uncensored)':
                gt = np.asarray(validation_data_full[p])[:, None]  # (n_data, 1)
            else:
                gt = np.asarray(validation_data[p])[:, None]  # (n_data, 1)
            err = np.sqrt(np.mean((ps - gt) ** 2, axis=0))
            normalizer = gt.max(axis=0) - gt.min(axis=0)
            err = err.flatten() / normalizer.flatten()
            param_errors.append(err)

        pos = base_positions + offsets[i]

        for ax in (ax_top, ax_bottom):
            bp = ax.boxplot(
                param_errors,
                positions=pos,
                widths=0.15,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black"),
                boxprops=dict(facecolor=color, alpha=0.7),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
            )
            if ax is ax_top:
                bp["boxes"][0].set_label(labels[i])

    # ---- define break limits ----
    ax_bottom.set_ylim(0, 0.4)
    ax_top.set_ylim(1.5, ax_top.get_ylim()[1])

    # hide spines between axes
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    # diagonal break marks
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_bottom.set_xticks(base_positions)
    ax_bottom.set_xticklabels(param_names_pretty)
    ax_bottom.set_ylabel("NRMSE")
    ax_top.legend(frameon=False)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return


# fig, ax = plt.subplots(figsize=(5, 2.5), layout="constrained")
#
# base_positions = np.arange(len(param_names))
# # symmetric offsets around each parameter index
# width = 0.2
# offsets = [-width, 0,  width]
# for i, ((model_name, samples), color) in enumerate(zip(posterior_samples_model.items(), colors_ordered)):
#     pos = base_positions + offsets[i]
#
#     # compute error per parameter
#     param_errors = []
#     for j, p in enumerate(param_names):
#         ps = np.asarray(samples[p])              # (n_data, n_samples)
#         if model_name == 'NPE (Uncensored)':
#             gt = np.asarray(validation_data_full[p])[:, None]
#         else:
#             gt = np.asarray(validation_data[p])[:, None]  # (n_data, 1)
#         err = np.sqrt(np.mean((ps - gt)**2, axis=0))
#         normalizer = gt.max(axis=0) - gt.min(axis=0)   # default bf
#         err = err.flatten()  / normalizer.flatten()
#         param_errors.append(err)
#
#     bp = ax.boxplot(
#         param_errors,
#         positions=pos,
#         widths=0.15,
#         patch_artist=True,
#         showfliers=False,
#         medianprops=dict(color="black"),
#         boxprops=dict(facecolor=color, alpha=0.7),
#         whiskerprops=dict(color=color),
#         capprops=dict(color=color),
#     )
#     bp["boxes"][0].set_label(labels[i])
#
# #ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
# ax.set_xticks(base_positions)
# ax.set_xticklabels(param_names_pretty)
# ax.set_ylabel("NRMSE")
# ax.legend(frameon=False)
# ax.set_ylim(0, 0.6)
# #fig.savefig(BASE / 'plots' / f'recovery_boxplot.pdf', bbox_inches='tight')
# plt.show()


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
    # Hazard rows (a01, a02, a12)
    for h_idx, a_name in enumerate(hazard_params):
        row = h_idx

        trans = a_name[-2:]  # '01', '02', '12'
        a_key, shape_key = weibull_param_map[trans]

        for col_idx, e in enumerate(epochs):
            this_ax = ax[row, col_idx]

            # Naive Cox reference
            # this_ax.axhline(
            #     baseline[e][a_name]['naive_cox'],
            #     color=colors[0],
            #     label='Naive Cox' if (row == 2 and col_idx == 0) else None,
            #     zorder=3,
            # )
            t_days = baseline[e][f'naive_cox_times_{trans}']
            t_years = np.asarray(t_days) / 365.25
            haz = baseline[e][f'naive_cox_{trans}']

            this_ax.plot(
                t_years,
                haz,
                color=colors[0],
                label='Naive Cox' if (row == 2 and col_idx == 0) else None,
                zorder=3, alpha=0.75,
            )

            # IDM curves
            for b_i, b in enumerate(['idm_weib']): #, 'idm_splines']):
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
                this_ax.set_ylabel(r'$h_{'+a_name[-2:]+'}$', fontsize=12)
            if row == 0:
                this_ax.set_title(f'Epoch {e[-1]}', fontsize=14)
            elif row == nrows - 1:
                this_ax.set_xlabel('Follow up years since entry in epoch')

            # remove top and right spines
            this_ax.spines['top'].set_visible(False)
            this_ax.spines['right'].set_visible(False)


    # ------------------------------------------------------------
    # Beta rows (age, sex)
    # ------------------------------------------------------------
    for row_id, beta in enumerate(beta_params):
        row_idx = row_id + 3  # after hazard rows
        for col_idx, e in enumerate(epochs):
            this_ax = ax[row_idx, col_idx]

            vals_naive = [baseline[e][f'beta{t}_{beta}']['naive_cox'] for t in transitions]
            vals_weib = [baseline[e][f'beta{t}_{beta}']['weibull'] for t in transitions]
            # vals_spl = [baseline[e][f'beta{t}_{beta}']['splines'] for t in transitions]

            x = np.arange(len(transitions))

            this_ax.plot(x, vals_naive, marker='o', color=colors[0], linestyle='dotted', zorder=3,
                         label='Naive Cox' if (row_idx == 0 and col_idx == 0) else None)
            this_ax.plot(x, vals_weib, marker='o', color=colors[1], linestyle='dotted', zorder=4, alpha=0.75,
                         label='Weibull IDM' if (row_idx == 0 and col_idx == 0) else None)
            # this_ax.plot(x, vals_spl, marker='o', color=colors[2], linestyle='dotted', zorder=4, alpha=0.75,
            #              label='Splines IDM' if (row_idx == 0 and col_idx == 0) else None)

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
                this_ax.set_ylabel(beta.replace('_', ' '), fontsize=12)
            if row_idx == 3:
                this_ax.set_xlabel('Transition', fontsize=12)

            # remove top and right spines
            this_ax.spines['top'].set_visible(False)
            this_ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------
    # Global legend
    legend_handles = [
        Patch(facecolor=colors[0], label='Naive Cox'),
        Patch(facecolor=colors[1], alpha=0.75, label='Weibull IDM'),
        # Patch(facecolor=colors[2],  alpha=0.75, label='Splines IDM'),
        Patch(facecolor='black', alpha=0.15, label='Prior 95% interval'),
        Patch(facecolor=colors[3], alpha=0.3, label='Posterior 95% CI'),
        Line2D([0], [0], color=colors[3], linestyle='--', label='Posterior median'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncols=5,
        frameon=False,
        fontsize=12,
    )

    if save_path is not None:
        fig.savefig(save_path / f'{network_name}_params.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return


def plot_cumhaz(baseline, posterior_samples, df_real, network_name, transition='all', per_person=100, adjust_cov=True, save_path=None):
    n_epochs = len(epochs)
    if transition == 'all':
        transition = ['01', '02', '12']
    else:
        transition = [transition]
    fig, axes_all = plt.subplots(len(transition), n_epochs, figsize=(1.75 * n_epochs, 3 * len(transition)), layout='constrained',
                            sharey='row', sharex=True)
    for trans_i, trans in enumerate(transition):
        if len(transition) > 1:
            axes = axes_all[trans_i]
        else:
            axes = axes_all
        # Handle case of single epoch
        if n_epochs == 1:
            axes = [axes]

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
        naive_trend = []
        weib_trend = []
        npe_trend = []
        for epoch_idx, e in enumerate(epochs):
            ax = axes[epoch_idx]

            # subset data for this epoch
            df_e = df_real[df_real['epoch'] == e]
            ages = df_e['age'].to_numpy()
            ages_mean = np.mean(ages)
            sexs = df_e['sex'].to_numpy()
            sexs_mean = np.mean(sexs)

            # ------------------------------------------------------------------
            # 1) Age/sex-adjusted point curves: Naive Cox, Weibull IDM, Splines IDM
            # ------------------------------------------------------------------
            # Naive Cox
            lp_naive = (
                    baseline[e][age_coef_name]['naive_cox'] * ages_mean +
                    baseline[e][sex_coef_name]['naive_cox'] * sexs_mean
            )
            w_bar_naive = np.exp(lp_naive)

            # Weibull IDM
            lp_weib = (
                    baseline[e][age_coef_name]['weibull'] * ages_mean +
                    baseline[e][sex_coef_name]['weibull'] * sexs_mean
            )
            w_bar_weib = np.exp(lp_weib)

            # # Splines IDM
            # lp_spl = (
            #         baseline[e][age_coef_name]['splines'] * ages_mean +
            #         baseline[e][sex_coef_name]['splines'] * sexs_mean
            # )
            # w_bar_spl = np.exp(lp_spl)

            # Weibull IDM baseline hazard and cumulative hazard (epoch-local)
            t_weib = np.asarray(baseline[e]['idm_weib_times'])
            h_weib_base = np.asarray(baseline[e][f'idm_weib_{trans}'])
            if adjust_cov:
                h_weib_adj = h_weib_base * w_bar_weib
            else:
                h_weib_adj = h_weib_base
            h_weib_local = per_person * cumulative_trapz(t_weib, h_weib_adj)  # starts at 0

            # # Splines IDM
            # t_spl = np.asarray(baseline[e]['idm_splines_times'])
            # h_spl_base = np.asarray(baseline[e][f'idm_splines_{trans}'])
            # if adjust_cov:
            #     h_spl_adj = h_spl_base * w_bar_spl
            # else:
            #     h_spl_adj = h_spl_base
            # H_spl_local = per_person * cumulative_trapz(t_spl, h_spl_adj)

            # Naive Cox a01
            t_cox = np.asarray(baseline[e][f'naive_cox_times_{trans}'])
            h_cox_base = np.asarray(baseline[e][f'naive_cox_{trans}'])
            if adjust_cov:
                h_cox_adj = h_cox_base * w_bar_naive
            else:
                h_cox_adj = h_cox_base
            h_naive_local = per_person * cumulative_trapz(t_cox, h_cox_adj)  # starts at 0

            # h_naive_base = baseline[e][a01_param_name]['naive_cox']
            # if adjust_cov:
            #     h_naive_adj = h_naive_base * w_bar_naive
            # else:
            #     h_naive_adj = h_naive_base
            # t_naive = t_weib
            # H_naive_local = per_person * cumulative_trapz(
            #     t_naive,
            #     np.full_like(t_naive, h_naive_adj, dtype=float),
            # )

            # Convert to years for plotting
            t_naive_years = (t_cox - t_cox[0]) / 365
            t_weib_years = (t_weib - t_weib[0]) / 365
            # t_spl_years = (t_spl - t_spl[0]) / 365

            # Plot point estimates
            ax.plot(t_naive_years, h_naive_local, color=colors[0],
                    label='Naive Cox' if epoch_idx == 0 else None)
            naive_trend.append(h_naive_local[-1])
            ax.plot(t_weib_years, h_weib_local, color=colors[1],
                    label='Weibull IDM' if epoch_idx == 0 else None)
            weib_trend.append(h_weib_local[-1])
            # ax.plot(t_spl_years, H_spl_local, color=colors[2],
            #         label='Splines IDM' if epoch_idx == 0 else None)
            #logging.info(f"Epoch {e}: H(5y) splines={H_spl_local[-1]:.4g} per {per_person}")

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

                h_base_post_samples = np.empty((n_samp_post, t_grid_e.size))
                for k in range(n_samp_post):
                    lp_k = beta_age_post_epoch[k] * ages_mean + beta_sex_post_epoch[k] * sexs_mean
                    w_bar_k = np.exp(lp_k)

                    h_k = weibull_hazard(t_grid_e, a_samp_epoch[k], shape_samp_epoch[k])
                    if adjust_cov:
                        h_base_post_samples[k, :] = h_k * w_bar_k
                    else:
                        h_base_post_samples[k, :] = h_k

                h_post_samples = per_person * cumulative_trapz_samples(t_grid_e, h_base_post_samples)
                med_e = np.median(h_post_samples, axis=0)
                low_e = np.quantile(h_post_samples, 0.025, axis=0)
                high_e = np.quantile(h_post_samples, 0.975, axis=0)
                npe_trend.append(h_post_samples[:, -1])

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
                if trans == '01':
                    label = 'Dementia hazard\n'
                elif trans == '02':
                    label = 'Death hazard\n'
                else:
                    label = 'Dementia/death hazard\n'
                ax.set_ylabel(f'{label} per {per_person} persons', fontsize=12)

            # remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlim(0, 5)
            ax.set_ylim(0, None)

        fig.supxlabel('Follow up years since entry in epoch', fontsize=12)
        # Single legend below the figure
        legend_handles = [
            Patch(facecolor=colors[0], label='Naive Cox'),
            Patch(facecolor=colors[1], alpha=0.75, label='Weibull IDM'),
            # Patch(facecolor=colors[2],  alpha=0.75, label='Splines IDM'),
            Patch(facecolor=colors[3], alpha=0.3, label='Posterior 95% CI'),
            Line2D([0], [0], color=colors[3], linestyle='--', label='Posterior median'),
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False,
                   fontsize=12, ncols=2)

        # print trends at 5 years for each method
        x = np.arange(len(epochs))
        logging.info(f"Trends at 5 years (per {per_person}):")
        logging.info(f'Naive Cox: {np.polyfit(x, naive_trend, 1)[0]}')
        logging.info(f'Weibull IDM: {np.polyfit(x, weib_trend, 1)[0]}')
        if has_posterior:
            trend = []
            for i in range(npe_trend[0].shape[0]):
                y = [trend_epoch[i] for trend_epoch in npe_trend]
                trend.append(np.polyfit(x, y, 1)[0])
            logging.info(f'Posterior: {np.median(trend)} (95% CI {np.quantile(trend, 0.025)}, {np.quantile(trend, 0.975)})')

    if save_path is not None:
        if len(transition) == 3:
            transition = 'all'
        else:
            transition = transition[0]
        plt.savefig(save_path / f'{network_name}_{transition}_cumhaz_age_sex.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return


def plot_hazard_nrmse(
    posterior_samples_model, validation_data, validation_data_full,
    labels_dict, save_path=None
):

    # --- helpers ---
    def adjusted_a(a, beta_sex, beta_age, mean_sex, mean_age):
        # multiplicative covariate effect on the scale
        return a * np.exp(beta_sex * mean_sex + beta_age * mean_age)

    def hazard_from_params(t_grid, a, shape, beta_sex, beta_age, mean_sex, mean_age):
        a_adj = adjusted_a(a, beta_sex, beta_age, mean_sex, mean_age)
        return weibull_hazard(t_grid[None, :], a_adj[:, None], shape[:, None])  # (n, T)

    def nrmse(x, y):
        diff2 = (x - y) ** 2
        rmse = np.sqrt(np.mean(diff2))
        denom = np.mean(np.abs(y))
        return rmse / denom

    # --- choose covariate means (average individual) ---
    mean_sex = float(np.mean(validation_data["sex"]))
    mean_age = float(np.mean(validation_data["age"]))

    # --- choose a time grid for evaluating hazards ---
    times = np.concatenate([np.asarray(validation_data["dt"]).ravel(),
                            np.asarray(validation_data["ds"]).ravel()])
    times = times[np.isfinite(times)]
    t_max = float(np.quantile(times, 0.95)) if times.size else 10.0
    t_min = max(1e-3, float(np.quantile(times, 0.05))) if times.size else 1e-3
    t_grid = np.linspace(t_min, t_max, 80)

    # --- mapping for the three transitions ---
    transitions = ["01", "02", "12"]
    hazard_labels = [r"$h_{01}$", r"$h_{02}$", r"$h_{12}$"]

    # --- compute ground-truth hazards (average individual), per datapoint ---
    # validation_data contains *true* params per datapoint, so compute h_true for each i
    h_true = {}
    for tr in transitions:
        a_key = f"a{tr}"
        s_key = f"shape{tr}"
        bs_key = f"beta{tr}_sex"
        ba_key = f"beta{tr}_age"

        a = np.asarray(validation_data[a_key]).ravel()
        s = np.asarray(validation_data[s_key]).ravel()
        bs = np.asarray(validation_data[bs_key]).ravel()
        ba = np.asarray(validation_data[ba_key]).ravel()

        h_true[tr] = hazard_from_params(t_grid, a, s, bs, ba, mean_sex, mean_age)  # (n_data, T)

    h_true_full = {}
    for tr in transitions:
        a_key = f"a{tr}"
        s_key = f"shape{tr}"
        bs_key = f"beta{tr}_sex"
        ba_key = f"beta{tr}_age"

        a = np.asarray(validation_data_full[a_key]).ravel()
        s = np.asarray(validation_data_full[s_key]).ravel()
        bs = np.asarray(validation_data_full[bs_key]).ravel()
        ba = np.asarray(validation_data_full[ba_key]).ravel()

        h_true_full[tr] = hazard_from_params(t_grid, a, s, bs, ba, mean_sex, mean_age)  # (n_data, T)

    # --- now compute per-model posterior hazard NRMSE distributions ---
    hazard_errors_model = {}  # model -> list of 3 arrays (each: (n_draws,))
    for model_name, samples in posterior_samples_model.items():
        errs_per_tr = []

        for tr in transitions:
            a_key = f"a{tr}"
            s_key = f"shape{tr}"
            bs_key = f"beta{tr}_sex"
            ba_key = f"beta{tr}_age"

            # expected shapes:
            # samples[key]: (n_data, n_draws)
            a = np.asarray(samples[a_key])[:, :, 0]
            s = np.asarray(samples[s_key])[:, :, 0]
            bs = np.asarray(samples[bs_key])[:, :, 0]
            ba = np.asarray(samples[ba_key])[:, :, 0]

            n_data, n_draws = a.shape

            # compute hazards for each draw:
            tr_errs = np.empty(n_draws, dtype=float)

            if 'uncensored' in model_name:
                hT = h_true_full[tr]  # (n_data, T)
            else:
                hT = h_true[tr]  # (n_data, T)

            for d in range(n_draws):
                hD = hazard_from_params(
                    t_grid,
                    a[:, d], s[:, d], bs[:, d], ba[:, d],
                    mean_sex, mean_age
                )  # (n_data, T)

                # NRMSE over all elements (data x time)
                tr_errs[d] = nrmse(hD, hT)

            errs_per_tr.append(tr_errs)

        hazard_errors_model[model_name] = errs_per_tr

    # --- plot: grouped boxplot for h01,h02,h12 across models ---
    fig, ax = plt.subplots(figsize=(5, 2.5), layout="constrained")

    base_positions = np.arange(len(transitions))
    width = 0.2
    offsets = (np.arange(len(posterior_samples_model)) - (len(posterior_samples_model) - 1) / 2) * width

    for i, (model_name, _) in enumerate(posterior_samples_model.items()):
        label = labels_dict[model_name][0]
        color = labels_dict[model_name][1]

        param_errors = hazard_errors_model[model_name]  # [errs01, errs02, errs12], each (n_draws,)
        pos = base_positions + offsets[i]

        bp = ax.boxplot(
            param_errors,
            positions=pos,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black"),
            boxprops=dict(facecolor=color, alpha=0.7),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        bp["boxes"][0].set_label(label)

    ax.set_xticks(base_positions)
    ax.set_xticklabels(hazard_labels)
    ax.set_ylabel("Hazard NRMSE")
    ax.legend(facecolor='white', framealpha=1, edgecolor='white', fancybox=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')
    #ax.set_ylim(0, 0.1)
    ax.set_yscale('log')
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_data_summaries(df_real, save_path=None):
    base_color = colors[0]
    dementia_color = "#E3B23C"
    death_color = "#FF5A5E"
    inconclusive_color = colors[-1]

    df = df_real.copy()
    case = df["case"].astype(str)

    # --- mid state (inconclusive is mid; dementia mid if dementia-related; else healthy mid) ---
    mid_I = case.eq("inconclusive")
    mid_D = case.isin(["dementia only", "died following dementia"])
    mid_H = ~(mid_I | mid_D)

    # --- end state (Healthy / Dementia / Death); inconclusive -> Death (per your requirement) ---
    end_H = case.eq("alive and dementia-free")
    end_D = case.eq("dementia only")
    end_X = case.isin(["died dementia-free", "died following dementia"]) | case.eq("inconclusive")

    def count(mid_mask, end_mask):
        return int((mid_mask & end_mask).sum())

    # Start -> Mid
    n_S_H = int(mid_H.sum())
    n_S_D = int(mid_D.sum())
    n_S_I = int(mid_I.sum())

    # Mid Healthy -> End
    n_HH = count(mid_H, end_H)
    n_HD = count(mid_H, end_D)  # expected ~0 by construction
    n_HX = count(mid_H, end_X)

    # Mid Dementia -> End
    n_DH = count(mid_D, end_H)  # expected 0
    n_DD = count(mid_D, end_D)
    n_DX = count(mid_D, end_X)

    # Mid Inconclusive -> End (forced to Death)
    n_IH = 0
    n_ID = 0
    n_IX = n_S_I

    labels = [
        "Healthy",
        " ",
        "Dementia",
        "Censored",
        "Healthy",
        "Dementia",
        "Death",
    ]

    node_colors = [
        base_color,
        base_color,
        dementia_color,
        inconclusive_color,
        base_color,
        dementia_color,
        death_color,
    ]

    # Links
    source = [
        0, 0, 0,  # Start -> Mid
        1, 1, 1,  # Mid Healthy -> End
        2, 2, 2,  # Mid Dementia -> End
        3, 3, 3  # Mid Inconclusive -> End
    ]
    target = [
        1, 2, 3,
        4, 5, 6,
        4, 5, 6,
        4, 5, 6
    ]
    value = [
        n_S_H, n_S_D, n_S_I,
        n_HH, n_HD, n_HX,
        n_DH, n_DD, n_DX,
        n_IH, n_ID, n_IX
    ]

    link_colors = [
        "rgba(75,46,131,0.25)",  # Start->Mid Healthy
        "rgba(227,178,60,0.25)",  # Start->Mid Dementia
        "rgba(27,138,143,0.30)",  # Start->Mid Inconclusive

        "rgba(75,46,131,0.20)",  # MidH->EndH
        "rgba(227,178,60,0.20)",  # MidH->EndD
        "rgba(255,90,94,0.25)",  # MidH->EndX

        "rgba(75,46,131,0.20)",  # MidD->EndH
        "rgba(227,178,60,0.25)",  # MidD->EndD
        "rgba(255,90,94,0.25)",  # MidD->EndX

        "rgba(75,46,131,0.20)",  # MidI->EndH
        "rgba(227,178,60,0.20)",  # MidI->EndD
        "rgba(255,90,94,0.30)",  # MidI->EndX (forced)
    ]

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=18,
                line=dict(width=0),
                label=labels,
                color=node_colors,
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
            ),
        )]
    )

    # --- publication styling ---
    fig.update_layout(
        width=900,
        height=420,
        font=dict(
            family="CMU Serif, Computer Modern, Latin Modern Roman, Times New Roman, serif",
            size=30,
            color="black"
        ),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # remove interactive title for paper figure
    fig.update_layout(title=None)

    # --- export as vector PDF ---
    if save_path is not None:
        fig.write_image(save_path, scale=2)
    fig.show()

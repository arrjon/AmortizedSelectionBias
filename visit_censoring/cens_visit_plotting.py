import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import cumulative_trapezoid
import plotly.graph_objects as go


colors = ['#4B2E83', '#D64A62', '#E68600', '#1B8A8F']
idm_param_map = {
    '01': ('a01', 'shape01'),
    '02': ('a02', 'shape02'),
    '12': ('a12', 'shape12'),
}
epochs = ["epoch1", "epoch2", "epoch3", "epoch4"]
params_beta = ['beta01_age', 'beta02_age', 'beta12_age', 'beta01_sex', 'beta02_sex', 'beta12_sex']
params_a = ['a01', 'a02', 'a12']
params_shape = ['shape01', 'shape02', 'shape12']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def cumulative_trapz_samples(t, h_samples):
    return cumulative_trapezoid(h_samples, x=t, axis=1, initial=0.0)


def weibull_hazard(t, a, shape):
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
                gt = np.asarray(validation_data_full[p])[:, None]
            else:
                gt = np.asarray(validation_data[p])[:, None]
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

    ax_bottom.set_ylim(0, 0.4)
    ax_top.set_ylim(1.5, ax_top.get_ylim()[1])

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

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


def plot_params(
    baseline,
    prior_samples,
    prior_summary,
    posterior_samples,
    posterior_summary,
    network_name,
    posterior_samples_2=None,
    posterior_summary_2=None,
    network_name_2=None,
    show_cox=True,
    idm_model='spl',
    save_path=None,
):
    transitions = ['01', '02', '12']
    beta_params = ['age', 'sex']
    hazard_params = ['a01', 'a02', 'a12']
    idm_key = 'idm_splines' if idm_model == 'spl' else 'idm_weib'
    idm_label = 'Splines IDM' if idm_model == 'spl' else 'Weibull IDM'

    npe_color = colors[3] if 'full' not in network_name else colors[2]
    npe_name = 'Bias-aware NPE' if 'full' not in network_name else 'NPE'
    has_second = posterior_samples_2 is not None and network_name_2 is not None
    npe_color_2 = (colors[3] if 'full' not in network_name_2 else colors[2]) if has_second else None
    npe_name_2 = ('Bias-aware NPE' if 'full' not in network_name_2 else 'NPE') if has_second else None

    n_epochs = len(epochs)
    fig, ax = plt.subplots(
        nrows=2 + len(hazard_params),
        ncols=n_epochs,
        figsize=(3 * n_epochs, 2 * (2 + len(hazard_params))),
        layout='constrained',
        sharey='row',
    )

    # ------------------------------------------------------------
    # Hazard rows (a01, a02, a12)
    for h_idx, a_name in enumerate(hazard_params):
        row = h_idx
        trans = a_name[-2:]
        a_key, shape_key = idm_param_map[trans]

        for col_idx, e in enumerate(epochs):
            this_ax = ax[row, col_idx]
            lc = row == 2 and col_idx == 0

            if show_cox:
                cox_h = baseline[e][f'H{trans}']['cox']
                this_ax.plot(
                    np.asarray(cox_h['haz_time']) / 365,
                    np.asarray(cox_h['haz']),
                    color=colors[0],
                    label='Naive Cox' if lc else None,
                    zorder=3, alpha=0.75,
                )

            t_days = baseline[e][f'H{trans}'][idm_key]['times']
            haz_idm = baseline[e][f'H{trans}'][idm_key]['cumhaz']
            this_ax.plot(
                np.asarray(t_days) / 365,
                haz_idm,
                color=colors[1],
                label=idm_label if lc else None,
                zorder=4, alpha=0.75,
            )

            times_all = np.array(baseline[e]['H01'][idm_key]['times'])
            t_grid = np.linspace(1e-3, times_all.max(), 200)
            t_grid_years = t_grid / 365

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
                    color='black', alpha=0.15,
                    label='Prior 95% interval' if lc else None,
                    zorder=1,
                )

            for ps, ps_color, ps_name in [
                (posterior_samples,   npe_color,   npe_name),
                (posterior_samples_2, npe_color_2, npe_name_2),
            ]:
                if ps is None or a_key not in ps or shape_key not in ps:
                    continue
                a_s = ps[a_key][col_idx].flatten()
                s_s = ps[shape_key][col_idx].flatten()
                haz_samp = weibull_hazard(t_grid[None, :], a_s[:, None], s_s[:, None])
                this_ax.fill_between(
                    t_grid_years,
                    np.quantile(haz_samp, 0.025, axis=0),
                    np.quantile(haz_samp, 0.975, axis=0),
                    color=ps_color, alpha=0.3,
                    label=f'{ps_name} posterior 95% CI' if lc else None,
                    zorder=2,
                )
                this_ax.plot(
                    t_grid_years,
                    np.median(haz_samp, axis=0),
                    linestyle='--', color=ps_color,
                    label=f'{ps_name} posterior median' if lc else None,
                    zorder=5,
                )

            this_ax.set_yscale('log')
            if col_idx == 0:
                this_ax.set_ylabel(r'$h_{' + a_name[-2:] + '}$', fontsize=12)
            if row == 0:
                this_ax.set_title(f'Epoch {e[-1]}', fontsize=14)
            elif row == 2:
                this_ax.set_xlabel('Follow up years since entry in epoch')
            this_ax.spines['top'].set_visible(False)
            this_ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------
    # Beta rows (age, sex)
    # ------------------------------------------------------------
    idm_coef_key = 'splines' if idm_model == 'spl' else 'weibull'
    for row_id, beta in enumerate(beta_params):
        row_idx = row_id + 3
        for col_idx, e in enumerate(epochs):
            this_ax = ax[row_idx, col_idx]
            lc = row_idx == 3 and col_idx == 0

            vals_naive = [baseline[e][f'beta{t}_{beta}']['naive_cox'] for t in transitions]
            vals_idm   = [baseline[e][f'beta{t}_{beta}'][idm_coef_key] for t in transitions]
            x = np.arange(len(transitions))

            if show_cox:
                this_ax.plot(x, vals_naive, marker='o', color=colors[0], zorder=3,
                             label='Naive Cox' if lc else None)
            this_ax.plot(x, vals_idm, marker='o', color=colors[1], zorder=4, alpha=0.75,
                         label=idm_label if lc else None)

            q = [prior_summary[f'beta{t}_{beta}'] for t in transitions]
            if q:
                this_ax.fill_between(
                    x,
                    [q[i]['low'] for i in range(len(transitions))],
                    [q[i]['high'] for i in range(len(transitions))],
                    color='black', alpha=0.15,
                    label='Prior 95% interval' if lc else None,
                    zorder=1,
                )

            for ps_sum, ps_color, ps_name, zo in [
                (posterior_summary,   npe_color,   npe_name,   3),
                (posterior_summary_2, npe_color_2, npe_name_2, 2),
            ]:
                if ps_sum is None:
                    continue
                q2 = [ps_sum[f'beta{t}_{beta}'] for t in transitions]
                if not q2:
                    continue
                this_ax.fill_between(
                    x,
                    [q2[i]['low'][col_idx] for i in range(len(transitions))],
                    [q2[i]['high'][col_idx] for i in range(len(transitions))],
                    color=ps_color, alpha=0.3,
                    label=f'{ps_name} posterior 95% CI' if lc else None,
                    zorder=zo,
                )
                this_ax.plot(
                    x,
                    [q2[i]['median'][col_idx] for i in range(len(transitions))],
                    linestyle='--', color=ps_color,
                    label=f'{ps_name} posterior median' if lc else None,
                    zorder=5,
                )

            this_ax.set_xticks(x)
            this_ax.set_xticklabels(transitions)
            if col_idx == 0:
                this_ax.set_ylabel(beta.replace('_', ' '), fontsize=12)
            if row_idx == 4:
                this_ax.set_xlabel('Transition', fontsize=12)
            this_ax.spines['top'].set_visible(False)
            this_ax.spines['right'].set_visible(False)

    # ------------------------------------------------------------
    # Global legend
    legend_handles = []
    if show_cox:
        legend_handles.append(Patch(facecolor=colors[0], label='Naive Cox'))
    legend_handles += [
        Patch(facecolor=colors[1], alpha=0.75, label=idm_label),
        Patch(facecolor='black', alpha=0.15, label='Prior'),
        Patch(facecolor=npe_color, alpha=0.3, label=npe_name),
    ]
    if has_second:
        legend_handles.append(Patch(facecolor=npe_color_2, alpha=0.3, label=npe_name_2))

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize=12,
        ncols=len(legend_handles),
    )

    if save_path is not None:
        fig.savefig(save_path / f'{network_name}_params.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_cumhaz(baseline, posterior_samples, df_real, network_name,
                posterior_samples_2=None, network_name_2=None,
                prior_samples=None,
                transition='all', per_person=100, adjust_cov=True,
                show_cox=True, idm_model='spl', save_path=None):
    n_epochs = len(epochs)
    if transition == 'all':
        transition = ['01', '02', '12']
    else:
        transition = [transition]

    npe_color = colors[3] if 'full' not in network_name else colors[2]
    npe_name = r'Bias-aware NPE' if 'full' not in network_name else r'NPE'
    has_second = posterior_samples_2 is not None and network_name_2 is not None
    npe_color_2 = (colors[3] if 'full' not in network_name_2 else colors[2]) if has_second else None
    npe_name_2 = (r'Bias-aware NPE' if 'full' not in network_name_2 else r'NPE (observed data)') if has_second else None

    fig, axes_all = plt.subplots(
        len(transition), n_epochs,
        figsize=(1.75 * n_epochs, 3.1 * len(transition)),
        layout='constrained', sharey='row', sharex=True,
    )

    for trans_i, trans in enumerate(transition):
        axes = axes_all[trans_i] if len(transition) > 1 else axes_all
        if n_epochs == 1:
            axes = [axes]

        age_coef_name = f'beta{trans}_age'
        sex_coef_name = f'beta{trans}_sex'
        a_key, shape_key = idm_param_map[trans]

        epoch_lengths = {e: np.asarray(baseline[e]['H01']['spl']['time'])[-1] -
                            np.asarray(baseline[e]['H01']['spl']['time'])[0]
                         for e in epochs}
        t_local = np.linspace(1e-3, max(epoch_lengths.values()), 200)

        def _has_posterior(ps):
            return (ps is not None
                    and a_key in ps and shape_key in ps
                    and age_coef_name in ps and sex_coef_name in ps)

        has_posterior   = _has_posterior(posterior_samples)
        has_posterior_2 = _has_posterior(posterior_samples_2)
        has_prior = prior_samples is not None and a_key in prior_samples and shape_key in prior_samples

        for epoch_idx, e in enumerate(epochs):
            ax = axes[epoch_idx]

            df_e = df_real[df_real['epoch'] == e]
            ages_mean = np.mean(df_e['age'].to_numpy())
            sexs_mean = np.mean(df_e['sex'].to_numpy())

            # MLE curves
            if show_cox:
                b = baseline[e]['H' + trans]['cox']
                naive_cumhaz = np.array(b['cumhaz']) / 100 * per_person
                ax.plot(np.array(b['time']), naive_cumhaz, color=colors[0],
                        label='Naive Cox' if epoch_idx == 0 else None)
                ax.fill_between(np.array(b['ci_time']),
                                np.array(b['ci_low']) / 100 * per_person,
                                np.array(b['ci_high']) / 100 * per_person,
                                alpha=0.2, color=colors[0])

            b = baseline[e]['H' + trans][idm_model]
            idm_cumhaz = np.array(b['cumhaz']) / 100 * per_person
            ax.plot(np.array(b['time']), idm_cumhaz, color=colors[1],
                    label='IDM' if epoch_idx == 0 else None)
            ax.fill_between(np.array(b['ci_time']),
                            np.array(b['ci_low']) / 100 * per_person,
                            np.array(b['ci_high']) / 100 * per_person,
                            alpha=0.2, color=colors[1])

            # Time grid clipped to epoch length
            t_grid_e = t_local[t_local <= epoch_lengths[e]] * 365

            def _plot_posterior(ps, ps_color, ps_name):
                a_samp  = np.asarray(ps[a_key][epoch_idx]).flatten()
                s_samp  = np.asarray(ps[shape_key][epoch_idx]).flatten()
                ba_samp = np.asarray(ps[age_coef_name][epoch_idx]).flatten()
                bs_samp = np.asarray(ps[sex_coef_name][epoch_idx]).flatten()
                h_base = np.empty((a_samp.shape[0], t_grid_e.size))
                for k in range(a_samp.shape[0]):
                    h_k = weibull_hazard(t_grid_e, a_samp[k], s_samp[k])
                    h_base[k] = h_k * np.exp(ba_samp[k] * ages_mean + bs_samp[k] * sexs_mean) if adjust_cov else h_k
                h_cum = per_person * cumulative_trapz_samples(t_grid_e, h_base)
                ax.plot(t_grid_e / 365, np.median(h_cum, axis=0), linestyle='--', color=ps_color,
                        label=f'{ps_name} posterior median' if epoch_idx == 0 else None)
                ax.fill_between(t_grid_e / 365,
                                np.quantile(h_cum, 0.025, axis=0),
                                np.quantile(h_cum, 0.975, axis=0),
                                alpha=0.3, color=ps_color,
                                label=f'{ps_name} posterior 95% CI' if epoch_idx == 0 else None)

            if has_posterior:
                _plot_posterior(posterior_samples, npe_color, npe_name)
            if has_posterior_2:
                _plot_posterior(posterior_samples_2, npe_color_2, npe_name_2)

            # Prior drawn last to avoid inflating axis limits
            if has_prior:
                ylim = ax.get_ylim()
                a_pr = np.asarray(prior_samples[a_key]).flatten()
                s_pr = np.asarray(prior_samples[shape_key]).flatten()
                h_pr = np.empty((a_pr.shape[0], t_grid_e.size))
                for k in range(a_pr.shape[0]):
                    h_pr[k] = weibull_hazard(t_grid_e, a_pr[k], s_pr[k])
                h_pr_cum = per_person * cumulative_trapz_samples(t_grid_e, h_pr)
                ax.fill_between(
                    t_grid_e / 365,
                    np.quantile(h_pr_cum, 0.025, axis=0),
                    np.quantile(h_pr_cum, 0.975, axis=0),
                    color='black', alpha=0.15,
                    label='Prior 95% interval' if epoch_idx == 0 else None,
                    zorder=1,
                )
                ax.set_ylim(ylim)

            if trans_i == 0:
                ax.set_title(f'Epoch {e[-1]}', fontsize=18)
            if epoch_idx == 0:
                label = {'01': 'Dementia', '02': 'Death', '12': 'Dementia/death'}[trans]
                ax.set_ylabel(f'{label} cumulative\n hazard per {per_person} persons', fontsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.set_xlim(0, 5)
            ax.set_ylim(0.1, None)
            ax.set_yscale('log')

        fig.supxlabel(r'Follow up years since entry in epoch', fontsize=20)
        legend_handles = []
        if has_prior:
            legend_handles.append(Patch(facecolor='black', alpha=0.15, label='Prior'))
        if show_cox:
            legend_handles.append(Patch(facecolor=colors[0], label='Naive Cox'))
        #else:
        #    legend_handles.append(Patch(facecolor=colors[0], label='NPE (full data)'))
        if has_posterior_2:
            legend_handles.append(Patch(facecolor=npe_color_2, alpha=0.3, label=npe_name_2))
        if has_posterior:
            legend_handles.append(Patch(facecolor=npe_color, alpha=0.3, label=npe_name))
        legend_handles.append(Patch(facecolor=colors[1], alpha=0.75, label=r'IDM'))
        fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
                   bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=20)

    if save_path is not None:
        tr_str = 'all' if len(transition) == 3 else transition[0]
        plt.savefig(save_path / f'{network_name}_{tr_str}_cumhaz_age_sex.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_hazard_nrmse(
    posterior_samples_model, validation_data, validation_data_full,
    labels_dict, save_path=None
):
    def adjusted_a(a, beta_sex, beta_age, mean_sex, mean_age):
        return a * np.exp(beta_sex * mean_sex + beta_age * mean_age)

    def hazard_from_params(t_grid, a, shape, beta_sex, beta_age, mean_sex, mean_age):
        a_adj = adjusted_a(a, beta_sex, beta_age, mean_sex, mean_age)
        return weibull_hazard(t_grid[None, :], a_adj[:, None], shape[:, None])

    mean_sex = float(np.mean(validation_data["sex"]))
    mean_age = float(np.mean(validation_data["age"]))

    times = np.concatenate([np.asarray(validation_data["dt"]).ravel(),
                            np.asarray(validation_data["ds"]).ravel()])
    times = times[np.isfinite(times)]
    t_max = float(np.quantile(times, 0.95)) if times.size else 10.0
    t_min = max(1e-3, float(np.quantile(times, 0.05))) if times.size else 1e-3
    t_grid = np.linspace(t_min, t_max, 80)

    transitions = ["01", "02", "12"]
    hazard_labels = [r"$h_{01}$", r"$h_{02}$", r"$h_{12}$"]

    def _true_hazards(data):
        result = {}
        for tr in transitions:
            a   = np.asarray(data[f"a{tr}"]).ravel()
            s   = np.asarray(data[f"shape{tr}"]).ravel()
            bs  = np.asarray(data[f"beta{tr}_sex"]).ravel()
            ba  = np.asarray(data[f"beta{tr}_age"]).ravel()
            result[tr] = hazard_from_params(t_grid, a, s, bs, ba, mean_sex, mean_age)
        return result

    h_true      = _true_hazards(validation_data)
    h_true_full = _true_hazards(validation_data_full)

    hazard_errors_model = {}
    for model_name, samples in posterior_samples_model.items():
        errs_per_tr = []
        for tr in transitions:
            a  = np.asarray(samples[f"a{tr}"])[:, :, 0]
            s  = np.asarray(samples[f"shape{tr}"])[:, :, 0]
            bs = np.asarray(samples[f"beta{tr}_sex"])[:, :, 0]
            ba = np.asarray(samples[f"beta{tr}_age"])[:, :, 0]
            n_data, n_draws = a.shape
            ht = h_true_full[tr] if 'uncensored' in model_name else h_true[tr]
            tr_errs = np.empty((n_draws, n_data), dtype=float)
            for d in range(n_draws):
                hd = hazard_from_params(t_grid, a[:, d], s[:, d], bs[:, d], ba[:, d], mean_sex, mean_age)
                diff = hd - ht  # n_data, n_time
                tr_errs[d] = np.sqrt(np.mean(diff ** 2, axis=-1)) / np.mean(ht, axis=-1) # n_data
            errs_per_tr.append(np.median(tr_errs, axis=0))
        hazard_errors_model[model_name] = errs_per_tr

    fig, ax = plt.subplots(figsize=(5, 2.5), layout="constrained")
    base_positions = np.arange(len(transitions))
    width = 0.2
    offsets = (np.arange(len(posterior_samples_model)) - (len(posterior_samples_model) - 1) / 2) * width

    for i, (model_name, _) in enumerate(posterior_samples_model.items()):
        label = labels_dict[model_name][0]
        color = labels_dict[model_name][1]
        pos = base_positions + offsets[i]
        bp = ax.boxplot(
            hazard_errors_model[model_name],
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
    ax.set_xticklabels(hazard_labels, fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylabel("Hazard NRMSE", fontsize=14)
    #ax.legend(facecolor='white', framealpha=1, edgecolor='white', fancybox=False, fontsize=14,
    #          loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')
    ax.set_yscale('log')
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_data_summaries(df_real, save_path=None):
    base_color = '#5B8DB8'  # steel blue
    dementia_color = '#E07B54'  # terracotta
    death_color = '#888888'  # mid grey
    inconclusive_color = '#A86BAD'  # muted purple-lilac

    link_colors = [
        "rgba(91,141,184,0.25)",  # Start->Mid Healthy
        "rgba(224,123,84,0.25)",  # Start->Mid Dementia
        "rgba(168,107,173,0.30)",  # Start->Mid Inconclusive
        "rgba(91,141,184,0.20)",  # MidH->EndH
        "rgba(224,123,84,0.20)",  # MidH->EndD
        "rgba(136,136,136,0.25)",  # MidH->EndX
        "rgba(91,141,184,0.20)",  # MidD->EndH
        "rgba(224,123,84,0.25)",  # MidD->EndD
        "rgba(136,136,136,0.25)",  # MidD->EndX
        "rgba(91,141,184,0.20)",  # MidI->EndH
        "rgba(224,123,84,0.20)",  # MidI->EndD
        "rgba(136,136,136,0.30)",  # MidI->EndX (forced)
    ]

    df = df_real.copy()
    case = df["case"].astype(str)

    mid_I = case.eq("inconclusive")
    mid_D = case.isin(["dementia only", "died following dementia"])
    mid_H = ~(mid_I | mid_D)

    end_H = case.eq("alive and dementia-free")
    end_D = case.eq("dementia only")
    end_X = case.isin(["died dementia-free", "died following dementia"]) | case.eq("inconclusive")

    def count(mid_mask, end_mask):
        return int((mid_mask & end_mask).sum())

    n_S_H = int(mid_H.sum())
    n_S_D = int(mid_D.sum())
    n_S_I = int(mid_I.sum())

    n_HH = count(mid_H, end_H)
    n_HD = count(mid_H, end_D)
    n_HX = count(mid_H, end_X)
    n_DH = count(mid_D, end_H)
    n_DD = count(mid_D, end_D)
    n_DX = count(mid_D, end_X)

    node_colors = [
        base_color, base_color, dementia_color, inconclusive_color,
        base_color, dementia_color, death_color,
    ]
    source = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    target = [1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6]
    value  = [n_S_H, n_S_D, n_S_I, n_HH, n_HD, n_HX, n_DH, n_DD, n_DX, 0, 0, n_S_I]

    fig = go.Figure(
        data=[go.Sankey(
            arrangement="snap",
            node=dict(pad=18, thickness=18, line=dict(width=0), color=node_colors),
            link=dict(source=source, target=target, value=value, color=link_colors),
        )]
    )
    fig.update_layout(
        width=960, height=360,
        font=dict(
            family="CMU Serif, Computer Modern, Latin Modern Roman, Times New Roman, serif",
            size=30, color="black",
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        title=None,
    )
    if save_path is not None:
        fig.write_image(save_path, scale=2)
    fig.show()

import numpy as np
import matplotlib.pyplot as plt

colors = ['#4a2377', '#f55f74', '#f47a00', '#0d7d87']
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
    """Single curve cumulative integral."""
    t = np.asarray(t)
    h = np.asarray(h)
    dt = np.diff(t)
    H = np.zeros_like(h)
    H[1:] = np.cumsum(0.5 * (h[1:] + h[:-1]) * dt)
    return H

def cumulative_trapz_samples(t, h_samples):
    """
    Cumulative integral for an array of curves.
    h_samples: shape (n_samples, n_time)
    """
    t = np.asarray(t)
    h_samples = np.asarray(h_samples)
    dt = np.diff(t)  # (n_time-1,)
    H = np.zeros_like(h_samples)
    H[:, 1:] = np.cumsum(
        0.5 * (h_samples[:, 1:] + h_samples[:, :-1]) * dt[None, :],
        axis=1,
    )
    return H

def weibull_hazard(t, a, shape):
    # h(t) = a * shape * t^(shape - 1)
    t = np.asarray(t, dtype=float)
    return a * shape * np.power(t, shape - 1)


def plot_hazard(baseline, prior_samples, prior_summary, posterior_samples, posterior_summary, network_name):
    n_beta = len(params_beta)
    n_epochs = len(epochs)
    n_a = len(params_a)

    # total panels: beta coefficients + hazard panels (3 per epoch)
    total_panels = n_beta + n_a * n_epochs
    ncols = 3
    nrows = int(np.ceil(total_panels / ncols))

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8, 10),
        layout='constrained',
        sharey='row',
    )
    ax = ax.flatten()

    # ------------------------------------------------------------------
    # 1) Beta panels
    # ------------------------------------------------------------------
    for i, p in enumerate(params_beta):
        this_ax = ax[i]

        vals_naive = [baseline[e][p]['naive_cox'] for e in epochs]
        vals_weib = [baseline[e][p]['weibull'] for e in epochs]
        vals_splines = [baseline[e][p]['splines'] for e in epochs]

        this_ax.plot(epochs, vals_naive, label='Naive Cox',
                     marker='o', color=colors[0])
        this_ax.plot(epochs, vals_weib, label='Weibull IDM',
                     marker='o', color=colors[1])
        this_ax.plot(epochs, vals_splines, label='Splines IDM',
                     marker='o', color=colors[2])

        this_ax.set_title(p)

        xmin, xmax = this_ax.get_xlim()

        if p in prior_summary:
            q = prior_summary[p]
            # Prior band
            this_ax.fill_between(
                [xmin, xmax],
                [q["low"], q["low"]],
                [q["high"], q["high"]],
                alpha=0.15,
                color='black',
                label='Prior 95% interval' if i == 0 else None,
            )

        if p in posterior_summary:
            low = posterior_summary[p]["low"]
            high = posterior_summary[p]["high"]
            median = posterior_summary[p]["median"]
            # Posterior band
            this_ax.fill_between(
                epochs,
                low,
                high,
                alpha=0.15,
                color=colors[3],
                label='Posterior 95% interval' if i == 0 else None,
            )
            # Posterior median
            this_ax.plot(
                epochs,
                median,
                linestyle='--',
                color=colors[3],
                label='Posterior median' if i == 0 else None,
            )

    # ------------------------------------------------------------------
    # 2) Hazard panels
    # ------------------------------------------------------------------

    # Hazards start after the beta panels
    hazard_base_idx = n_beta

    # for legend labels of hazard priors/posteriors (only once)
    prior_haz_band_done = False
    prior_haz_med_done = False
    post_haz_band_done = False
    post_haz_med_done = False

    # Precompute naive Cox baseline per epoch & param_a
    naive_baselines = {
        (e, a_name): baseline[e][a_name]['naive_cox']
        for e in epochs
        for a_name in params_a
    }

    # One big loop over hazard panels indexed by (epoch_idx, a_idx)
    for epoch_idx, e in enumerate(epochs):
        for a_idx, a_name in enumerate(params_a):
            ax_idx = hazard_base_idx + epoch_idx * n_a + a_idx
            if ax_idx >= len(ax):
                continue  # safety guard if total_panels < nrows*ncols

            this_ax = ax[ax_idx]

            # Naive Cox reference line
            this_ax.axhline(naive_baselines[(e, a_name)],
                            color=colors[0],
                            label='Naive Cox' if (epoch_idx == 0 and a_idx == 0) else None)

            # IDM curves
            for b_i, b in enumerate(['idm_weib', 'idm_splines']):
                times = np.array([baseline[ep][f'{b}_times'] for ep in epochs]).T
                a01 = np.array([baseline[ep][f'{b}_01'] for ep in epochs]).T

                this_ax.plot(
                    times[:, epoch_idx],
                    a01[:, epoch_idx],
                    color=colors[b_i + 1],
                    label=('Weibull IDM' if b == 'idm_weib' else 'Splines IDM')
                    if (epoch_idx == 0 and a_idx == 0) else None
                )

            this_ax.set_xlabel('time')
            this_ax.set_title(f'{e}: {a_name}')
            this_ax.set_yscale('log')

            # --- Weibull prior hazard bands ---
            trans = ['01', '02', '12'][a_idx]  # mapping aligned with params_a
            a_key, shape_key = weibull_param_map[trans]

            # To define t_grid we need a sensible max time over epochs
            times_all = np.array([baseline[ep]['idm_weib_times'] for ep in epochs]).flatten()
            t_max = times_all.max()
            t_grid = np.linspace(1e-3, t_max, 200)

            if a_key in prior_samples and shape_key in prior_samples:
                a_samp = prior_samples[a_key]
                shape_samp = prior_samples[shape_key]

                # vectorised hazard evaluation: (n_samples, len(t_grid))
                haz_samples = weibull_hazard(t_grid[None, :],
                                             a_samp[:, None],
                                             shape_samp[:, None])
                haz_med = np.median(haz_samples, axis=0)
                haz_low = np.quantile(haz_samples, 0.025, axis=0)
                haz_high = np.quantile(haz_samples, 0.975, axis=0)

                this_ax.fill_between(
                    t_grid,
                    haz_low,
                    haz_high,
                    alpha=0.15,
                    color='black',
                    label='Prior 95% interval' if not prior_haz_band_done else None,
                )
                prior_haz_band_done = True

            # --- Weibull posterior hazard bands (per epoch) ---
            if a_key in posterior_samples and shape_key in posterior_samples:
                a_samp_all = posterior_samples[a_key]
                shape_samp_all = posterior_samples[shape_key]

                # take epoch-specific samples
                a_samp_epoch = a_samp_all[epoch_idx].flatten()
                shape_samp_epoch = shape_samp_all[epoch_idx].flatten()

                haz_samples = weibull_hazard(t_grid[None, :],
                                             a_samp_epoch[:, None],
                                             shape_samp_epoch[:, None])
                haz_med = np.median(haz_samples, axis=0)
                haz_low = np.quantile(haz_samples, 0.025, axis=0)
                haz_high = np.quantile(haz_samples, 0.975, axis=0)

                this_ax.fill_between(
                    t_grid,
                    haz_low,
                    haz_high,
                    alpha=0.15,
                    color=colors[3],
                    label='Posterior 95% interval' if not post_haz_band_done else None,
                )
                this_ax.plot(
                    t_grid,
                    haz_med,
                    linestyle='--',
                    color=colors[3],
                    label='Posterior median' if not post_haz_med_done else None,
                )
                post_haz_band_done = True
                post_haz_med_done = True

    # ------------------------------------------------------------------
    # 3) Global legend (unique labels)
    # ------------------------------------------------------------------
    handles, labels = [], []
    for a in ax:
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                labels.append(ll)
                handles.append(hh)

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.07),
        ncols=3
    )

    fig.savefig(f'plots/binder/binder_{network_name}_hazards.png', bbox_inches='tight')
    return fig

def plot_a1(baseline, prior_samples, posterior_samples, network_name):
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')

    # ----------------------------------------------------------------------
    # Precompute epoch-wise time grids and lengths
    # ----------------------------------------------------------------------
    epoch_times_weib = {e: np.asarray(baseline[e]['idm_weib_times']) for e in epochs}
    epoch_lengths = {e: t[-1] - t[0] for e, t in epoch_times_weib.items()}
    max_length = max(epoch_lengths.values())

    # Local time grid for hazard computation within an epoch
    t_local = np.linspace(1e-3, max_length, 200)

    # Weibull parameter keys for transition 01
    trans = '01'
    a_key, shape_key = weibull_param_map[trans]

    # ----------------------------------------------------------------------
    # Prior hazard band (same for all epochs, just shifted in time)
    # ----------------------------------------------------------------------
    prior_haz_med = prior_haz_low = prior_haz_high = None
    if a_key in prior_samples and shape_key in prior_samples:
        a_samp_prior = prior_samples[a_key]
        shape_samp_prior = prior_samples[shape_key]

        # Shape: (n_samples, len(t_local))
        haz_samples_prior = weibull_hazard(
            t_local[None, :],
            a_samp_prior[:, None],
            shape_samp_prior[:, None],
        )

        prior_haz_med = np.median(haz_samples_prior, axis=0)
        prior_haz_low = np.quantile(haz_samples_prior, 0.025, axis=0)
        prior_haz_high = np.quantile(haz_samples_prior, 0.975, axis=0)

    # ----------------------------------------------------------------------
    # Plot concatenated a01 for Naive Cox, Weibull IDM, Splines IDM
    # and overlay prior/posterior bands per epoch
    # ----------------------------------------------------------------------
    t_all_weib, a01_all_weib = [], []
    t_all_spl, a01_all_spl = [], []
    t_all_naive, a01_all_naive = [], []

    epoch_boundaries = []
    offset = 0.0

    prior_band_done = False
    post_band_done = False

    # Posterior samples (epoch-specific)
    has_posterior = (a_key in posterior_samples
                     and shape_key in posterior_samples)
    if has_posterior:
        a_samp_post_all = posterior_samples[a_key]
        shape_samp_post_all = posterior_samples[shape_key]

    # assumes params_a[0] corresponds to a01
    a01_param_name = params_a[0]

    for epoch_idx, e in enumerate(epochs):
        # ----- concatenate model curves -----
        t_weib = epoch_times_weib[e]
        a01_weib = np.asarray(baseline[e]['idm_weib_01'])

        t_spl = np.asarray(baseline[e]['idm_splines_times'])
        a01_spl = np.asarray(baseline[e]['idm_splines_01'])

        # shift to concatenated time
        t_weib_shift = (t_weib - t_weib[0]) + offset
        t_spl_shift = (t_spl - t_spl[0]) + offset

        t_all_weib.append(t_weib_shift)
        a01_all_weib.append(a01_weib)
        t_all_spl.append(t_spl_shift)
        a01_all_spl.append(a01_spl)

        naive_val = baseline[e][a01_param_name]['naive_cox']
        t_all_naive.append(t_weib_shift)
        a01_all_naive.append(np.full_like(t_weib_shift, naive_val, dtype=float))

        epoch_start = offset
        epoch_len = epoch_lengths[e]

        # update offset and boundaries
        offset = t_weib_shift[-1]
        epoch_boundaries.append(offset)

        # ----- prior hazard band for this epoch (if available) -----
        if prior_haz_med is not None:
            mask = t_local <= epoch_len
            t_grid_e = t_local[mask]
            med_e = prior_haz_med[mask]
            low_e = prior_haz_low[mask]
            high_e = prior_haz_high[mask]

            x_grid_shift = t_grid_e + epoch_start
            ax.fill_between(
                x_grid_shift,
                low_e,
                high_e,
                alpha=0.15,
                color='black',
                label='Prior 95% interval' if not prior_band_done else None,
            )
            prior_band_done = True

        # ----- posterior hazard band for this epoch (if available) -----
        if has_posterior:
            a_samp_epoch = a_samp_post_all[epoch_idx].flatten()
            shape_samp_epoch = shape_samp_post_all[epoch_idx].flatten()

            haz_samples_post = weibull_hazard(
                t_local[None, :],
                a_samp_epoch[:, None],
                shape_samp_epoch[:, None],
            )
            post_med = np.median(haz_samples_post, axis=0)
            post_low = np.quantile(haz_samples_post, 0.025, axis=0)
            post_high = np.quantile(haz_samples_post, 0.975, axis=0)

            mask = t_local <= epoch_len
            t_grid_e = t_local[mask]
            med_e = post_med[mask]
            low_e = post_low[mask]
            high_e = post_high[mask]

            x_grid_shift = t_grid_e + epoch_start
            ax.fill_between(
                x_grid_shift,
                low_e,
                high_e,
                alpha=0.15,
                color=colors[3],
                label='Posterior hazard 95% interval' if not post_band_done else None,
            )
            ax.plot(
                x_grid_shift,
                med_e,
                linestyle='--',
                color=colors[3],
                label='Posterior hazard median' if not post_band_done else None,
            )
            post_band_done = True

    # ----------------------------------------------------------------------
    # Final concatenation and plotting of model curves
    # ----------------------------------------------------------------------
    t_all_weib = np.concatenate(t_all_weib)
    a01_all_weib = np.concatenate(a01_all_weib)
    t_all_spl = np.concatenate(t_all_spl)
    a01_all_spl = np.concatenate(a01_all_spl)
    t_all_naive = np.concatenate(t_all_naive)
    a01_all_naive = np.concatenate(a01_all_naive)

    ax.plot(t_all_naive, a01_all_naive, color=colors[0], label='Naive Cox a01')
    ax.plot(t_all_weib, a01_all_weib, color=colors[1], label='Weibull IDM a01')
    ax.plot(t_all_spl, a01_all_spl, color=colors[2], label='Splines IDM a01')

    # epoch boundaries
    for b in epoch_boundaries[:-1]:
        ax.axvline(b, color='grey', alpha=0.3, linestyle=':')

    ax.set_xlabel('time (concatenated over epochs)')
    ax.set_ylabel(r'$a_{01}$ hazard')
    ax.set_yscale('log')
    ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.0))

    fig.savefig(f'plots/binder/binder_{network_name}_a01.png', bbox_inches='tight')
    return fig

def plot_a1_median(baseline, prior_samples, posterior_samples, network_name):
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')

    # ----------------------------------------------------------------------
    # Precompute epoch-wise time grids and lengths
    # ----------------------------------------------------------------------
    epoch_times_weib = {e: np.asarray(baseline[e]['idm_weib_times']) for e in epochs}
    epoch_lengths = {e: t[-1] - t[0] for e, t in epoch_times_weib.items()}
    max_length = max(epoch_lengths.values())

    # Local time grid for hazard computation within an epoch
    t_local = np.linspace(1e-3, max_length, 200)

    # Weibull parameter keys for transition 01
    trans = '01'
    a_key, shape_key = weibull_param_map[trans]

    # ----------------------------------------------------------------------
    # Prior hazard band (same for all epochs, just shifted in time)
    # ----------------------------------------------------------------------
    prior_haz_med = prior_haz_low = prior_haz_high = None
    if a_key in prior_samples and shape_key in prior_samples:
        a_samp_prior = prior_samples[a_key]
        shape_samp_prior = prior_samples[shape_key]

        # Shape: (n_samples, len(t_local))
        haz_samples_prior = weibull_hazard(
            t_local[None, :],
            a_samp_prior[:, None],
            shape_samp_prior[:, None],
        )

        prior_haz_med = np.median(haz_samples_prior, axis=0)
        prior_haz_low = np.quantile(haz_samples_prior, 0.025, axis=0)
        prior_haz_high = np.quantile(haz_samples_prior, 0.975, axis=0)

    # ----------------------------------------------------------------------
    # Plot concatenated a01 for Naive Cox, Weibull IDM, Splines IDM
    # and overlay prior/posterior bands per epoch
    # ----------------------------------------------------------------------
    t_all_weib, a01_all_weib = [], []
    t_all_spl, a01_all_spl = [], []
    t_all_naive, a01_all_naive = [], []

    epoch_boundaries = []
    offset = 0.0

    prior_band_done = False
    post_band_done = False

    # Posterior samples (epoch-specific)
    has_posterior = (a_key in posterior_samples
                     and shape_key in posterior_samples)
    if has_posterior:
        a_samp_post_all = posterior_samples[a_key]
        shape_samp_post_all = posterior_samples[shape_key]

    # assumes params_a[0] corresponds to a01
    a01_param_name = params_a[0]

    for epoch_idx, e in enumerate(epochs):
        # ----- concatenate model curves -----
        t_weib = epoch_times_weib[e]
        a01_weib = np.asarray(baseline[e]['idm_weib_01'])

        t_spl = np.asarray(baseline[e]['idm_splines_times'])
        a01_spl = np.asarray(baseline[e]['idm_splines_01'])

        # shift to concatenated time
        t_weib_shift = (t_weib - t_weib[0]) + offset
        t_spl_shift = (t_spl - t_spl[0]) + offset

        t_all_weib.append(t_weib_shift)
        a01_all_weib.append(a01_weib)
        t_all_spl.append(t_spl_shift)
        a01_all_spl.append(a01_spl)

        naive_val = baseline[e][a01_param_name]['naive_cox']
        t_all_naive.append(t_weib_shift)
        a01_all_naive.append(np.full_like(t_weib_shift, naive_val, dtype=float))

        epoch_start = offset
        epoch_len = epoch_lengths[e]

        # update offset and boundaries
        offset = t_weib_shift[-1]
        epoch_boundaries.append(offset)

        # ----- prior hazard band for this epoch (if available) -----
        if prior_haz_med is not None:
            mask = t_local <= epoch_len
            t_grid_e = t_local[mask]
            med_e = prior_haz_med[mask]
            low_e = prior_haz_low[mask]
            high_e = prior_haz_high[mask]

            x_grid_shift = t_grid_e + epoch_start
            ax.fill_between(
                x_grid_shift,
                low_e,
                high_e,
                alpha=0.15,
                color='black',
                label='Prior 95% interval' if not prior_band_done else None,
            )
            prior_band_done = True

        # ----- posterior hazard band for this epoch (if available) -----
        if has_posterior:
            a_samp_epoch = a_samp_post_all[epoch_idx].flatten()
            shape_samp_epoch = shape_samp_post_all[epoch_idx].flatten()

            haz_samples_post = weibull_hazard(
                t_local[None, :],
                a_samp_epoch[:, None],
                shape_samp_epoch[:, None],
            )
            post_med = np.median(haz_samples_post, axis=0)
            post_low = np.quantile(haz_samples_post, 0.025, axis=0)
            post_high = np.quantile(haz_samples_post, 0.975, axis=0)

            mask = t_local <= epoch_len
            t_grid_e = t_local[mask]
            med_e = post_med[mask]
            low_e = post_low[mask]
            high_e = post_high[mask]

            x_grid_shift = t_grid_e + epoch_start
            ax.fill_between(
                x_grid_shift,
                np.median(low_e),
                np.median(high_e),
                alpha=0.15,
                color=colors[3],
                label='Posterior hazard 95% interval' if not post_band_done else None,
            )
            ax.plot(
                x_grid_shift,
                np.ones_like(med_e) * np.median(med_e),
                linestyle='--',
                color=colors[3],
                label='Posterior hazard median' if not post_band_done else None,
            )
            post_band_done = True

    # ----------------------------------------------------------------------
    # Final concatenation and plotting of model curves
    # ----------------------------------------------------------------------
    t_all_weib = np.concatenate(t_all_weib)
    a01_all_weib = np.concatenate(a01_all_weib)
    t_all_spl = np.concatenate(t_all_spl)
    a01_all_spl = np.concatenate(a01_all_spl)
    t_all_naive = np.concatenate(t_all_naive)
    a01_all_naive = np.concatenate(a01_all_naive)

    ax.plot(t_all_naive, a01_all_naive, color=colors[0], label='Naive Cox a01')
    ax.plot(t_all_weib, np.ones_like(a01_all_weib) * np.median(a01_all_weib), color=colors[1], label='Weibull Median a01')
    ax.plot(t_all_spl, np.ones_like(a01_all_spl) * np.median(a01_all_spl), color=colors[2], label='Splines Median a01')

    # epoch boundaries
    for b in epoch_boundaries[:-1]:
        ax.axvline(b, color='grey', alpha=0.3, linestyle=':')

    ax.set_xlabel('time (concatenated over epochs)')
    ax.set_ylabel(r'$a_{01}$ hazard')
    ax.set_yscale('log')
    ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.0))

    fig.savefig(f'plots/binder/binder_{network_name}_a01_median.png', bbox_inches='tight')
    return fig

def plot_cumhaz(baseline, posterior_samples, df_real, network_name):
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')

    age_coef_name  = 'beta01_age'
    sex_coef_name  = 'beta01_sex'
    a01_param_name = 'a01'          # Naive Cox baseline hazard key for 0→1

    # Weibull parameter names for transition 0→1
    trans = '01'
    a_key, shape_key = weibull_param_map[trans]

    # ----------------------------------------------------------------------
    # Time grid for prior/posterior cumulative hazards (epoch-local)
    # ----------------------------------------------------------------------
    epoch_lengths = {}
    for e in epochs:
        t_weib_e = np.asarray(baseline[e]['idm_weib_times'])
        epoch_lengths[e] = t_weib_e[-1] - t_weib_e[0]

    max_length = max(epoch_lengths.values())
    t_local = np.linspace(1e-3, max_length, 200)  # start slightly >0

    # check posterior availability
    has_posterior = (
        a_key in posterior_samples
        and shape_key in posterior_samples
        and age_coef_name in posterior_samples
        and sex_coef_name in posterior_samples
    )

    # flags so legend entries appear only once
    prior_band_done = False
    post_band_done = False

    # ----------------------------------------------------------------------
    # Age/sex–adjusted cumulative hazards for each model, per epoch
    # Epochs concatenated on x-axis, but *no* cumulative carry-over across epochs
    # ----------------------------------------------------------------------
    t_all_weib, H_all_weib = [], []
    t_all_spl,  H_all_spl  = [], []
    t_all_naive, H_all_naive = [], []

    epoch_boundaries = []
    time_offset = 0.0  # only for x-axis concatenation

    for epoch_idx, e in enumerate(epochs):
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
        w_bar_naive = np.exp(lp_naive).mean()

        # Weibull IDM
        lp_weib = (
            baseline[e][age_coef_name]['weibull'] * ages +
            baseline[e][sex_coef_name]['weibull'] * sexs
        )
        w_bar_weib = np.exp(lp_weib).mean()

        # Splines IDM
        lp_spl = (
            baseline[e][age_coef_name]['splines'] * ages +
            baseline[e][sex_coef_name]['splines'] * sexs
        )
        w_bar_spl = np.exp(lp_spl).mean()

        # Weibull IDM baseline hazard and cumulative hazard (epoch-local)
        t_weib = np.asarray(baseline[e]['idm_weib_times'])
        h_weib_base = np.asarray(baseline[e]['idm_weib_01'])
        h_weib_adj = h_weib_base * w_bar_weib
        H_weib_local = cumulative_trapz(t_weib, h_weib_adj)  # starts at 0

        # Splines IDM
        t_spl = np.asarray(baseline[e]['idm_splines_times'])
        h_spl_base = np.asarray(baseline[e]['idm_splines_01'])
        h_spl_adj = h_spl_base * w_bar_spl
        H_spl_local = cumulative_trapz(t_spl, h_spl_adj)

        # Naive Cox a01 (piecewise constant over epoch)
        h_naive_base = baseline[e][a01_param_name]['naive_cox']
        h_naive_adj = h_naive_base * w_bar_naive

        t_naive = t_weib
        H_naive_local = cumulative_trapz(
            t_naive,
            np.full_like(t_naive, h_naive_adj, dtype=float),
        )

        # concatenate in time, but do NOT sum cumulative hazards across epochs
        t_weib_shift  = (t_weib  - t_weib[0])  + time_offset
        t_spl_shift   = (t_spl   - t_spl[0])   + time_offset
        t_naive_shift = (t_naive - t_naive[0]) + time_offset

        t_all_weib.append(t_weib_shift)
        H_all_weib.append(H_weib_local)

        t_all_spl.append(t_spl_shift)
        H_all_spl.append(H_spl_local)

        t_all_naive.append(t_naive_shift)
        H_all_naive.append(H_naive_local)

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
                w_bar_k = np.exp(lp_k).mean()

                h_k = weibull_hazard(t_grid_e, a_samp_epoch[k], shape_samp_epoch[k])
                h_post_samples[k, :] = h_k * w_bar_k

            H_post_samples = cumulative_trapz_samples(t_grid_e, h_post_samples)
            med_e = np.median(H_post_samples, axis=0)
            low_e = np.quantile(H_post_samples, 0.025, axis=0)
            high_e = np.quantile(H_post_samples, 0.975, axis=0)

            x_grid_shift = t_grid_e + time_offset
            ax.fill_between(
                x_grid_shift,
                low_e,
                high_e,
                alpha=0.15,
                color=colors[3],
                label='Posterior cum. haz 95% interval' if not post_band_done else None,
            )
            ax.plot(
                x_grid_shift,
                med_e,
                linestyle='--',
                color=colors[3],
                label='Posterior cum. haz median' if not post_band_done else None,
            )
            post_band_done = True

        # update x-axis offset (no carry-over of H)
        time_offset = t_weib_shift[-1]
        epoch_boundaries.append(time_offset)

    # ----------------------------------------------------------------------
    # Concatenate age/sex–adjusted model cumulative hazards and plot
    # ----------------------------------------------------------------------
    t_all_weib   = np.concatenate(t_all_weib)
    H_all_weib   = np.concatenate(H_all_weib)
    t_all_spl    = np.concatenate(t_all_spl)
    H_all_spl    = np.concatenate(H_all_spl)
    t_all_naive  = np.concatenate(t_all_naive)
    H_all_naive  = np.concatenate(H_all_naive)

    ax.plot(t_all_naive, H_all_naive, color=colors[0],
            label='Naive Cox (cum a01, age/sex adj)')
    ax.plot(t_all_weib,  H_all_weib,  color=colors[1],
            label='Weibull IDM (cum a01, age/sex adj)')
    ax.plot(t_all_spl,   H_all_spl,   color=colors[2],
            label='Splines IDM (cum a01, age/sex adj)')

    # epoch boundaries
    for b in epoch_boundaries[:-1]:
        ax.axvline(b, color='grey', alpha=0.3, linestyle=':')

    ax.set_xlabel('time (epochs concatenated)')
    ax.set_ylabel(r'cumulative hazard $H_{01}(t \mid \text{age, sex})$')
    ax.legend(loc='upper right', bbox_to_anchor=(1.7, 1.0))

    fig.savefig(f'plots/binder/binder_{network_name}_{a01_param_name}_cumhaz_age_sex.png', bbox_inches='tight')
    return fig

import numpy as np
import re


def clean_dict_keys(d):
    """Rename keys with dots to underscores."""
    rename = {}
    for k in list(d.keys()):
        if '.' in k and 'beta' in k:
            new_k = re.sub(r'\.', '_', k)
            rename[k] = new_k
    d.update({rename[k]: d.pop(k) for k in rename})
    return d


def extract_batches_to_dict(df, scheme=None, epoch=None, sample_scheme_per_rep_epoch=False, seed=None, max_indiv=2299):
    """
    Dict of numpy arrays shaped (n_batches, n_features).
    Batch is unique (replicate, scheme, epoch).
    Feature is per id slot within a batch.
    Patient fields are padded with zeros to max_indiv.
    Params are shaped (n_batches, 1).
    """
    if seed is not None:
        np.random.seed(seed)
    d = df.copy()
    if scheme is not None:
        d = d.loc[d['scheme'] == scheme]
    if epoch is not None:
        d = d.loc[d['epoch'] == epoch]
    if d.empty:
        raise ValueError("No data after filtering")

    if scheme is None and sample_scheme_per_rep_epoch:
        if d.empty:
            raise ValueError("No data to sample from")
        chosen = (
            d[["replicate", "epoch", "scheme"]]
            .drop_duplicates()
            .groupby(["replicate", "epoch"], sort=False)["scheme"]
            .apply(lambda s: np.random.choice(s.to_numpy()))
            .reset_index()
        )
        d = d.merge(chosen, on=["replicate", "epoch", "scheme"], how="inner")

    d = d.sort_values(['replicate', 'scheme', 'epoch', 'id']).reset_index(drop=True)

    patient_cols = ['id', 'illt', 'ills', 'dt', 'ds', 'sex', 'age', 'age_raw']
    param_cols = [
        'a01', 'a02', 'a12',
        'shape01', 'shape02', 'shape12',
        'beta01.sex', 'beta02.sex', 'beta12.sex',
        'beta01.age', 'beta02.age', 'beta12.age'
    ]
    meta_cols_as_features = ['replicate', 'scheme', 'epoch']
    gb = d.groupby(['replicate', 'scheme', 'epoch'], sort=True, group_keys=False)

    # check one row per id in each batch
    sizes = gb.size().to_numpy()
    ids_unique = gb['id'].nunique().to_numpy()
    if not np.all(sizes == ids_unique):
        raise ValueError("Each batch must have at most one row per id")

    batch_keys = list(gb.groups.keys())
    n_batches = len(batch_keys)

    scheme_name = {
        'Full': 0,
        'CensVisit': 1,
        'CensDeath': 2
    }

    if max_indiv <= 0:
        raise ValueError("max_indiv must be positive")

    def pad_array(values, length):
        out = -np.ones(length, dtype=np.float32)
        n = len(values)
        out[:n] = np.asarray(values, dtype=np.float32)
        return out

    def stack_patient(col):
        if col not in d.columns:
            return None
        mats = []
        for key in batch_keys:
            g = gb.get_group(key).sort_values('id')
            mats.append(pad_array(g[col].to_numpy(), max_indiv))
        return np.vstack(mats)  # (n_batches, max_indiv)

    def stack_meta(col):
        mats = []
        for key in batch_keys:
            rep, sch, ep = key
            if col == 'replicate':
                val = int(rep)
            elif col == 'scheme':
                val = scheme_name.get(sch, -1)
            elif col == 'epoch':
                val = float(ep[-1])
            else:
                continue

            arr = -np.ones(max_indiv, dtype=np.float32)
            n = gb.get_group(key)['id'].nunique()
            arr[:n] = val
            mats.append(arr)
        return np.vstack(mats)  # (n_batches, max_indiv)

    def stack_param(col):
        if col not in d.columns:
            return None
        vals = []
        for key in batch_keys:
            g = gb.get_group(key)
            vals.append(g[col].iloc[0])
        arr = np.asarray(vals, dtype=np.float32)
        return arr.reshape(n_batches, 1)

    result = {}

    # patient level arrays
    for col in patient_cols:
        arr = stack_patient(col)
        if arr is not None:
            result[col] = arr

    # meta as features
    for col in meta_cols_as_features:
        result[col] = stack_meta(col)

    # params per batch
    for col in param_cols:
        arr = stack_param(col)
        if arr is not None:
            result[col] = arr
    return clean_dict_keys(result)


def compute_gamma_params(mean, cv):
    """
    Compute gamma distribution parameters from mean and coefficient of variation.

    Parameters:
    -----------
    mean : float
        Desired mean of the gamma distribution
    cv : float
        Coefficient of variation (SD/mean), default=0.3 for moderate uncertainty

    Returns:
    --------
    dict with 'alpha' (shape) and 'beta' (rate) parameters
    """
    alpha = 1 / (cv ** 2)
    beta = alpha / mean
    return {'alpha': alpha, 'beta': beta}



def summarize_batch_dict(data, taus=(365, 730, 1825), time_scale=1825.0):
    """
    Compute informative survival / multi-state summary statistics for a batch dict.

    Expected keys:
      - 'ills' : (B, N) illness indicator (0/1), padded with -1
      - 'illt' : (B, N) illness time (days), padded with -1
      - 'ds'   : (B, N) death indicator (0/1), padded with -1
      - 'dt'   : (B, N) death time (days), padded with -1
    Optional:
      - 'age'  : (B, N), padded with -1
      - 'sex'  : (B, N), padded with -1

    Padding rule:
      Individuals with any padded value (-1) are excluded.

    Returns:
      new dict with per-batch summaries, shape (B, 1) or (B, k).
    """
    EPS = 1e-8

    out = data.copy()

    ills = data["ills"]
    ds   = data["ds"]
    illt = data["illt"]
    dt   = data["dt"]

    # valid individuals mask: padded values are -1
    valid = (ills != -1) & (ds != -1) & (illt != -1) & (dt != -1)

    def masked_mean(x):
        num = np.sum(x * valid, axis=-1, keepdims=True)
        den = np.sum(valid, axis=-1, keepdims=True)
        return num / np.maximum(den, EPS)

    def masked_sum(x):
        return np.sum(x * valid, axis=-1, keepdims=True)

    def safe_div(num, den):
        return num / np.maximum(den, EPS)

    # cast indicators
    ills01 = (ills == 1).astype(float)
    ds01   = (ds == 1).astype(float)

    # ---------- 1) Event probabilities ----------
    out["p_ill"]   = masked_mean(ills01)
    out["p_death"] = masked_mean(ds01)

    both      = ((ills == 1) & (ds == 1)).astype(float)
    death_noi = ((ds == 1) & (ills == 0)).astype(float)
    ill_nod   = ((ills == 1) & (ds == 0)).astype(float)
    none      = ((ills == 0) & (ds == 0)).astype(float)

    out["p_both"]        = masked_mean(both)
    out["p_death_noill"] = masked_mean(death_noi)
    out["p_ill_nodeath"] = masked_mean(ill_nod)
    out["p_none"]        = masked_mean(none)

    # ---------- 2) Conditional time-to-event summaries ----------
    ds_sum   = masked_sum(ds01)
    ills_sum = masked_sum(ills01)

    # Event-weighted means (p(event)*E[T|event])
    out["dt_weighted_mean"]   = masked_mean(ds01 * dt) / time_scale
    out["illt_weighted_mean"] = masked_mean(ills01 * illt) / time_scale

    # Conditional means given event
    out["dt_event_mean"] = safe_div(masked_sum(ds01 * dt), ds_sum) / time_scale
    out["illt_event_mean"] = safe_div(masked_sum(ills01 * illt), ills_sum) / time_scale

    # Conditional SD given event
    dt2 = safe_div(masked_sum(ds01 * (dt**2)), ds_sum) / (time_scale**2)
    it2 = safe_div(masked_sum(ills01 * (illt**2)), ills_sum) / (time_scale**2)

    out["dt_event_sd"]   = np.sqrt(np.maximum(dt2 - out["dt_event_mean"]**2, 0.0))
    out["illt_event_sd"] = np.sqrt(np.maximum(it2 - out["illt_event_mean"]**2, 0.0))

    # Illness before death probability + gap
    ill_before_death = ((ills == 1) & (ds == 1) & (illt < dt)).astype(float)
    out["p_ill_before_death"] = masked_mean(ill_before_death)

    gap = np.where((ills == 1) & (ds == 1) & valid, (dt - illt), 0.0)
    both_sum = masked_sum(both)

    out["gap_mean_if_both"] = safe_div(masked_sum(gap), both_sum) / time_scale

    # ---------- 3) Horizon-based incidence ----------
    for tau in taus:
        tau = float(tau)
        out[f"p_ill_by_{int(tau)}d"] = masked_mean(
            (((ills == 1) & (illt <= tau)).astype(float))
        )
        out[f"p_death_by_{int(tau)}d"] = masked_mean(
            (((ds == 1) & (dt <= tau)).astype(float))
        )

    # ---------- 4) Covariate contrasts ----------
    age = data["age"]
    age_valid = valid & (age != -1)

    def masked_mean_age(x):
        num = np.sum(x * age_valid, axis=-1, keepdims=True)
        den = np.sum(age_valid, axis=-1, keepdims=True)
        return num / np.maximum(den, EPS)

    age_c = age - masked_mean_age(age)
    ds_c  = ds01 - masked_mean(ds01)
    il_c  = ills01 - masked_mean(ills01)

    out["corr_age_death"] = safe_div(
        masked_mean_age(age_c * ds_c),
        np.sqrt(masked_mean_age(age_c**2) * masked_mean(ds_c**2) + EPS),
    )

    out["corr_age_ill"] = safe_div(
        masked_mean_age(age_c * il_c),
        np.sqrt(masked_mean_age(age_c**2) * masked_mean(il_c**2) + EPS),
    )

    sex = data["sex"]
    sex_valid = valid & (sex != -1)

    m = (sex == 1) & sex_valid
    f = (sex == 0) & sex_valid

    m_n = np.sum(m, axis=-1, keepdims=True)
    f_n = np.sum(f, axis=-1, keepdims=True)

    p_death_m = safe_div(np.sum(ds01 * m, axis=-1, keepdims=True), m_n)
    p_death_f = safe_div(np.sum(ds01 * f, axis=-1, keepdims=True), f_n)
    p_ill_m   = safe_div(np.sum(ills01 * m, axis=-1, keepdims=True), m_n)
    p_ill_f   = safe_div(np.sum(ills01 * f, axis=-1, keepdims=True), f_n)

    out["p_death_sex_diff"] = p_death_m - p_death_f
    out["p_ill_sex_diff"]   = p_ill_m - p_ill_f

    return out

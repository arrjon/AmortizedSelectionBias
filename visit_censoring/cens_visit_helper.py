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


def extract_batches_to_dict(df, scheme=None, epoch=None, seed=None, max_indiv=2299):
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

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


BASE = Path(__file__).resolve().parent

# Known population parameters of Munich (Pritsch et al., 2021)
age_probs = {'0-19': 0.168, '20-34': 0.25, '35-49': 0.223, '50-64': 0.187, '65-79': 0.118, '80+': 0.054}
sex_probs = {'m': 1-0.501, 'f': 0.501}
hh_size_probs = {'household_1': 0.549, 'household_2': 0.226, 'household_34': 0.124, 'household_5+': 0.101}
country_probs = {0: 1-0.305, 1: 0.305} # 0=Germany, 1=other
full_population_size = 1_561_720

test_specificity_params = {  # (from Olbricht et al., 2020)
    'specificity': 0.9972041,
    'sensitivity': 0.8860104
}


def make_priors(
    intercept_logodds: Tuple[float, float],
    coef_logor: Tuple[float, float],
    seed: int | None = None,
) -> Dict[str, float]:
    """
    Build priors for a logistic (odds-ratio) model with reference categories:
      - sex: male (ref)
      - age: 20-34 (ref)
      - birth_country: Germany (ref)
      - household_size: 1 (ref)

    Priors are specified on the log-odds (intercept) and log-odds-ratio (coefficients).
    """
    rng = np.random.default_rng(seed)

    def draw(mu_sd: Tuple[float, float]) -> float:
        mu, sd = mu_sd
        return rng.normal(mu, sd)

    return {
        "intercept": draw(intercept_logodds),

        "sex_female": draw(coef_logor),

        "age_0_19": draw(coef_logor),
        "age_35_49": draw(coef_logor),
        "age_50_64": draw(coef_logor),
        "age_65_79": draw(coef_logor),
        "age_gt_80": draw(coef_logor),

        "birth_country_others": draw(coef_logor),

        "household_size_2": draw(coef_logor),
        "household_size_3_4": draw(coef_logor),
        "household_size_gt_5": draw(coef_logor),
    }


def simulate_outcome(
    n: int,
    # covariates (can be scalars or length-n arrays)
    sex: str | np.ndarray,
    age: str | np.ndarray,
    birth_country: str | np.ndarray,
    household_size: str | np.ndarray,
    # parameters on log-odds scale
    intercept: float,
    sex_female: float,
    age_0_19: float,
    age_35_49: float,
    age_50_64: float,
    age_65_79: float,
    age_gt_80: float,
    bc_others: float,
    hh_2: float,
    hh_3_4: float,
    hh_gt_5: float,
    seed: int | None = None,
    return_prob: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Simulate binary outcome y ~ Bernoulli(sigmoid(eta)) from a logistic OR model.

    Reference categories (contribute 0):
      sex=male, age=20-34, birth_country=Germany, household_size=1

    Inputs can be scalars (applied to all n) or numpy arrays of length n.
    Returns dict with y, p, and eta (linear predictor).
    """
    rng = np.random.default_rng(seed)

    def to_array(x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            if x.shape[0] != n:
                raise ValueError(f"Expected array length {n}, got {x.shape[0]}")
            return x
        return np.full(n, x, dtype=object)

    sex_a = to_array(sex)
    age_a = to_array(age)
    bc_a = to_array(birth_country)
    hh_a = to_array(household_size)

    eta = np.full(n, intercept, dtype=float)

    # sex
    eta += (sex_a == "f") * float(sex_female)

    # age
    eta += (age_a == "0-19")  * age_0_19
    eta += (age_a == "35-49") * age_35_49
    eta += (age_a == "50-64") * age_50_64
    eta += (age_a == "65-79") * age_65_79
    eta += (age_a == "80+")   * age_gt_80
    # (age == "20-34") is reference => +0

    # country of birth
    eta += (bc_a == 1) * float(bc_others)

    # household size
    eta += (hh_a == "household_2")   * hh_2
    eta += (hh_a == "household_34")  * hh_3_4
    eta += (hh_a == "household_5+")  * hh_gt_5
    # (hh == "1") is reference => +0

    # logistic link
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(n=1, p=p, size=n).astype(int)

    out = {"y": y, "p": p, "eta": eta}
    if not return_prob:
        out.pop("p")
    return out


def simulate_test_results(
    y: np.ndarray,
    sensitivity: float,
    specificity: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate observed test results given true binary outcomes and test sensitivity/specificity.

    y: array of true binary outcomes (0/1)
    sensitivity: P(test=1 | true=1)
    specificity: P(test=0 | true=0)

    Returns array of observed test results (0/1).
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]

    test_results = np.where(
        y == 1,
        # for truly infected
        rng.binomial(1, sensitivity, size=n),
        # for truly uninfected
        rng.binomial(1, 1 - specificity, size=n)
    )
    return test_results


def rogan_gladen_correction(
        apparent_prevalence: float,
        sensitivity: float,
        specificity: float
) -> float:
    # apparent prevalence -> Rogan-Gladen correction (clipped to [0,1])
    denom = float(sensitivity + specificity - 1.0)
    if denom <= 0:
        prev_rg = np.nan
    else:
        prev_rg = np.clip((apparent_prevalence + specificity - 1.0) / denom, 0.0, 1.0)
    return prev_rg


def oversample(
    n_out: int,
    epoch_index: int,
    bootstrap: bool = False,
    max_rake_iters: int = 100,
    rake_tol: float = 1e-6,
    seed: int | None = None,
) -> Tuple[pd.DataFrame | None, pd.DataFrame]:
    """
    Oversample rows from original KoCo19 data to match Munich (or other) population targets.

    Match Munich marginal population targets (raking/IPF) if prior_oversample is None.

    Returns a resampled DataFrame of size n_out, sampled with replacement.
    If n_out equals the original data size, returns the original data.
    """

    rng = np.random.default_rng(seed)
    df = pd.read_csv(BASE / 'data' / f"koco19_T{epoch_index}_prepared.csv")
    n0 = len(df)

    # Expect columns: 'age_group', 'sex', 'hh_size', 'birth_country'
    required = ["age_group", "sex", "hh_size", "birth_country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if bootstrap:
        idx = rng.choice(np.arange(n0), size=n0, replace=True)
        df = df.iloc[idx].reset_index(drop=True)

    targets = {  # Munich population targets
        "age_group": age_probs,
        "sex": sex_probs,
        "hh_size": hh_size_probs,
        "birth_country": country_probs,
    }

    # Impute missing data via target proportions
    for col, probs in targets.items():
         miss = df[col].isna()
         if miss.any():
             df[col + '_filled'] = df[col].copy()
             df.loc[miss, col+'_filled'] = rng.choice(
                 list(probs.keys()),
                 size=miss.sum(),
                 p=list(probs.values())
             )

    if n_out == n0:
        return None, df
    elif n_out < n0:
        raise ValueError(f"n_out ({n_out}) must be larger than the original data size ({n0}).")
    weights = np.ones(n0, dtype=float)

    eps = 1e-12
    for rake_i in range(max_rake_iters):
        old = weights.copy()

        for col, target_probs in targets.items():
            # current weighted proportions for categories in target
            cur_tot = np.sum(weights) + eps
            cur = {}
            if col+'_filled' in df.columns:
                use_col = col+'_filled'
            else:
                use_col = col
            for cat in target_probs.keys():
                mask = (df[use_col] == cat)
                cur[cat] = np.sum(weights[mask]) / cur_tot

            # multiplicative adjustment per category
            for cat, tgt in target_probs.items():
                c = max(cur.get(cat, 0.0), eps)
                adj = tgt / c
                mask = (df[use_col] == cat)
                if np.any(mask):
                    weights[mask] *= adj

        # convergence check
        rel_change = np.max(np.abs(weights - old) / (np.abs(old) + eps))
        if rel_change < rake_tol:
            break
        #print(f"Raking converged in {rake_i+1} iterations (rel_change={rel_change:.6f})")

    # final normalize and sample
    weights = np.clip(weights, 0.0, np.inf)
    s = weights.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Sampling weights are degenerate; check priors/targets and data categories.")

    p = weights / s
    idx = rng.choice(np.arange(n0), size=n_out, replace=True, p=p)

    out = df.iloc[idx]
    out["oversample_prob"] = p[idx]
    out["original_pop_size"] = n0
    df["oversample_prob"] = p
    return out, df


def subselect_inverse(
    oversampled_pop: pd.DataFrame,
    n_target: int,
    selection_probs: np.ndarray,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Inverse of oversample() using importance weights.

    Oversample draws individuals with probability proportional to w_i.
    To invert, we downsample with probability proportional to 1 / w_i.

    Returns
    -------
    DataFrame of size n_target, sampled WITHOUT replacement,
    approximately restoring the original cohort distribution.
    """
    rng = np.random.default_rng(seed)

    n0 = len(oversampled_pop)
    if n_target > n0:
        raise ValueError(f"n_target ({n_target}) > oversampled size ({n0})")


    if np.any(selection_probs <= 0) or not np.all(np.isfinite(selection_probs)):
        raise ValueError("Invalid oversampling weights detected.")

    # inverse weights
    inv_w = 1.0 / selection_probs
    inv_p = inv_w / inv_w.sum()

    # sample without replacement using inverse probabilities
    idx = rng.choice(np.arange(n0), size=n_target, replace=False, p=inv_p)

    out = oversampled_pop.iloc[idx]
    out["subselect_prob"] = inv_p[idx]
    return out


def adjusted_prevalence(
    uncorrected_prev: np.ndarray,
    selection_probs: np.ndarray,
    sensitivity: float,
    specificity: float,
) -> Dict:
    """
    Compute adjusted prevalence from a subsample.
    """

    w = 1.0 / selection_probs

    if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("Invalid design weights.")

    apparent = np.sum(w * uncorrected_prev) / np.sum(w)
    adjusted = rogan_gladen_correction(
        apparent_prevalence=apparent,
        sensitivity=sensitivity,
        specificity=specificity,
    )

    return {
        "apparent_prevalence": apparent,
        "adjusted_prevalence": adjusted,
    }



def simulate_population(
    epoch_index: int,
    n_out: int,
    # priors for infection (logistic OR model)
    intercept_logodds: Tuple[float, float] = (-3.0, 1.0),
    coef_logor: Tuple[float, float] = (0.0, 0.5),
    use_real_outcomes: bool = False,
    bootstrap_resamples: int = 0,  #  0 is no bootstrap, >1 is number of bootstrap resamples
    seed: int | None = None,
) -> Dict:
    """
    Pipeline:
      1) Draw infection-model parameters from make_priors().
      2) Oversample KoCo rows to size n_out.
      3) Simulate true infection outcomes y via simulate_outcome().
      4) Simulate observed test results via simulate_test_results().
      5) Compute prevalence (true and observed) and return outputs.
    """
    if bootstrap_resamples > 0:
        bootstrap = True
    else:
        bootstrap = False
        bootstrap_resamples = 1
    out = []
    for b_i in range(bootstrap_resamples):
        # ---- oversampling ----

        # Oversample to construct a "realistic" population-like dataset
        pop_oversample, original_df = oversample(
            n_out=n_out,
            epoch_index=epoch_index,
            bootstrap=bootstrap,
            seed=None if seed is None else seed + 1 + b_i,
        )

        if not use_real_outcomes:
            # ---- infection-model parameters ----
            theta = make_priors(
                intercept_logodds=intercept_logodds,
                coef_logor=coef_logor,
                seed=None if seed is None else seed + 2 + b_i,
            )

            # ---- simulate true infection outcome ----
            sim = simulate_outcome(
                n=len(pop_oversample),
                sex=pop_oversample["sex"].values,
                age=pop_oversample["age_group"].values,
                birth_country=pop_oversample["birth_country"].values,
                household_size=pop_oversample["hh_size"].values,
                intercept=theta["intercept"],
                sex_female=theta["sex_female"],
                age_0_19=theta["age_0_19"],
                age_35_49=theta["age_35_49"],
                age_50_64=theta["age_50_64"],
                age_65_79=theta["age_65_79"],
                age_gt_80=theta["age_gt_80"],
                bc_others=theta["birth_country_others"],
                hh_2=theta["household_size_2"],
                hh_3_4=theta["household_size_3_4"],
                hh_gt_5=theta["household_size_gt_5"],
                seed=None if seed is None else seed + 3 + b_i,
                return_prob=True,
            )

            y_true = sim["y"].copy()

            # ---- simulate test results ----
            y_test = simulate_test_results(
                y=y_true,
                sensitivity=test_specificity_params["sensitivity"],
                specificity=test_specificity_params["specificity"],
                seed=None if seed is None else seed + 4 + b_i,
            )
            y_test = y_test.astype(float)
            y_test[pop_oversample['pos'].isna().values] = np.nan  # retain missingness from KoCo data

            # ---- prevalence estimates ----
            pop_oversample["y_true"] = y_true
            pop_oversample["y_test"] = y_test

            # ---- subselect inverse to original cohort size ----
            pop_out = subselect_inverse(  # just for creating an artificial subsample similar to KoCo19
                pop_oversample,
                n_target=pop_oversample["original_pop_size"].iloc[0],
                selection_probs=pop_oversample['oversample_prob'].values,
                seed=None if seed is None else seed + 5 + b_i,
            )
        else:
            # ---- use real outcomes from KoCo data ----
            pop_out = original_df
            inv_w = 1.0 / pop_out['oversample_prob']
            pop_out['subselect_prob'] = inv_w / inv_w.sum()

            pop_out["y_test"] = pop_out["pos"]
            pop_out["y_true"] = None  # unknown
            theta = None  # unknown


        prev_true = np.mean(y_true) if not use_real_outcomes else np.nan

        # complete cases only for test prevalence
        pop_out_nan_dropped = pop_out.dropna(subset=['y_test']).reset_index(drop=True)

        # ---- recompute test prevalence on subselected/real data ----
        prev_rg = rogan_gladen_correction(
            apparent_prevalence=np.mean(pop_out_nan_dropped["y_test"].values),
            sensitivity=test_specificity_params["sensitivity"],
            specificity=test_specificity_params["specificity"],
        )

        # ---- compute adjusted prevalence on subselected/real data ----
        adjusted = adjusted_prevalence(
            uncorrected_prev=pop_out_nan_dropped["y_test"].values,
            selection_probs=pop_out_nan_dropped['subselect_prob'].values,
            sensitivity=test_specificity_params["sensitivity"],
            specificity=test_specificity_params["specificity"],
        )

        out_dict = {
            "subsample": pop_out,
            "infection_params": theta,
            "prevalence_true": prev_true,
            "prevalence_subsample": prev_rg,
            "prevalence_subsample_weighted": adjusted["adjusted_prevalence"],
        }
        out.append(out_dict)

    if bootstrap:
        # aggregate bootstrap results
        prevalence_subsample_bs = np.array([o["prevalence_subsample"] for o in out])
        prevalence_subsample_weighted_bs = np.array([o["prevalence_subsample_weighted"] for o in out])

        return {
            "subsample": out[0]["subsample"],
            "infection_params": None,
            "prevalence_true": out[0]["prevalence_true"],
            "prevalence_subsample": prevalence_subsample_bs,
            "prevalence_subsample_weighted": prevalence_subsample_weighted_bs
        }
    else:
        return out[0]


if __name__ == "__main__":
    # Example usage
    for e_index in range(1, 6):
        print(f"\n--- Epoch T{e_index} ---")
        sim_out = simulate_population(
            epoch_index=e_index,
            n_out=int(full_population_size),
            use_real_outcomes=True,
            bootstrap_resamples=5
        )

        prevalence_subsample = np.mean(sim_out["prevalence_subsample"])
        prevalence_subsample_weighted = np.mean(sim_out["prevalence_subsample_weighted"])

        subsample_df = sim_out["subsample"]
        print(f"True prevalence: {sim_out['prevalence_true']*100:.2f}%")
        print(f"Prevalence (Subsample): {prevalence_subsample*100:.2f}%; Bias: {(prevalence_subsample-sim_out['prevalence_true'])*100:.2f}")
        print(f"Weighted prevalence: {prevalence_subsample_weighted*100:.2f}%; Bias: {(prevalence_subsample_weighted-sim_out['prevalence_true'])*100:.2f}")

    print("\n--- Simulation Study over all epochs ---")
    n_sims = 100
    errs = []
    errs_naive = []
    prevs = []
    for i in tqdm(range(n_sims)):
        sim_out = simulate_population(
            epoch_index=np.random.choice([1,2,3,4,5]),
            n_out=int(full_population_size * 0.1),
            seed=i,
        )
        true_prev  = sim_out["prevalence_true"]*100
        adj_prev   = sim_out["prevalence_subsample_weighted"]*100
        naive_prev = sim_out["prevalence_subsample"]*100

        # signed errors
        err_adj   = adj_prev - true_prev
        err_naive = naive_prev - true_prev
        errs.append(err_adj)
        errs_naive.append(err_naive)

        prevs.append(true_prev)


    def metrics(err):
        err = np.asarray(err)
        bias = err.mean()
        mse  = (err**2).mean()
        rmse = np.sqrt(mse)
        mae  = np.abs(err).mean()
        return bias, mse, rmse, mae


    bias_adj, mse_adj, rmse_adj, mae_adj = metrics(errs)
    bias_naive, mse_naive, rmse_naive, mae_naive = metrics(errs_naive)


    print("Weighted estimator:")
    print(f"  Bias  = {bias_adj:.4f}")
    print(f"  MAE   = {mae_adj:.4f}")
    print(f"  RMSE  = {rmse_adj:.4f}")

    print("\nNaive estimator:")
    print(f"  Bias  = {bias_naive:.4f}")
    print(f"  MAE   = {mae_naive:.4f}")
    print(f"  RMSE  = {rmse_naive:.4f}")

    plt.figure()
    plt.hist(np.array(errs_naive), bins=20, alpha=0.5, label='Naive Estimated Prevalence', color='orange')
    plt.axvline(bias_naive, alpha=0.5, color='orange')
    plt.hist(np.array(errs), bins=20, alpha=0.5, label='Weighted Estimated Prevalence', color='teal')
    plt.axvline(bias_adj, alpha=0.5, color='teal')
    plt.xlabel("Bias")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

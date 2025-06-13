import numpy as np
import pandas as pd
from scipy import stats

from PedCov.selection import data_selection
from PedCov.helper_functions import normalize_household_data


def simulate_outbreak(
    hh_size, duration_followup, age_cat, protected, p_asympto,
    inc_shape_asymp, inc_scale_asymp, inc_shape_symp, inc_scale_symp,
    kt_shape, kt_rate, delayDist, alpha, beta, delta,
    mu_inf_SI, mu_inf_SC, mu_inf_AI, mu_inf_AC, mu_inf_AA,
    mu_susc_I, mu_susc_C, mu_protect_acq, mu_protect_transm,
    dt=0.25
):
    # initialize
    is_index = np.zeros(hh_size, np.int8)
    is_inclu = np.zeros(hh_size, np.int8)
    incl_dt  = -np.ones(hh_size, np.float64)
    inf_date = np.full(hh_size, np.inf)
    infect_status = np.zeros(hh_size, np.int8)
    obs_time = np.full(hh_size, np.nan)

    idx = np.random.randint(hh_size)
    is_index[idx] = 1
    inf_date[idx] = 0.0
    if np.random.rand() < p_asympto:
        infect_status[idx] = 2
        # incubation = np.random.randint(minIncub, maxIncub+1), # note: does not work with STAN
        incubation = np.random.gamma(inc_shape_asymp, inc_scale_asymp)
    else:
        infect_status[idx] = 1
        incubation = np.random.gamma(inc_shape_symp, inc_scale_symp)
    obs_time[idx] = inf_date[idx] + incubation

    # precompute w
    w = (hh_size / 4.0)**(-delta)

    # main loop
    t = dt  # note: sophie did discrete steps with t=1
    while t <= duration_followup:
        susc = np.where(infect_status == 0)[0]
        if susc.size == 0:
            break
        infs = np.where(infect_status > 0)[0]

        # 1) compute per-infector multiplier
        inf_mult = np.empty(infs.size, np.float64)
        for k in range(infs.size):
            j = infs[k]
            age = age_cat[j]
            if infect_status[j] == 1:
                base = mu_inf_SI if age == 0 else (mu_inf_SC if age == 1 else 1.0)
            else:
                base = mu_inf_AI if age == 0 else (mu_inf_AC if age == 1 else mu_inf_AA)
            inf_mult[k] = base * (mu_protect_transm if protected[j] else 1.0)

        # 2) compute kernel(lag) vectorized
        lags = t - inf_date[infs]
        # zero out non‐positive lags
        for k in range(lags.size):
            if lags[k] <= 0:
                lags[k] = 0.0
        kernel_vals = stats.gamma.pdf(lags, kt_shape, scale=1.0/kt_rate)

        # 3) total transmission potential
        T = np.sum(inf_mult * kernel_vals)

        # 4) compute susceptibility multipliers
        susc_mult = np.empty(susc.size, np.float64)
        for k in range(susc.size):
            i = susc[k]
            age = age_cat[i]
            base = mu_susc_I if age == 0 else (mu_susc_C if age == 1 else 1.0)
            if protected[i]:
                base *= mu_protect_acq
            susc_mult[k] = base

        # 5) hazards & tau‐leap
        hazards = alpha + beta * w * susc_mult * T
        probs = 1.0 - np.exp(-hazards * dt)  # note: not done by sophie
        draws = np.random.rand(susc.size)
        new_inf_idx = np.where(draws < probs)[0]

        for ni in new_inf_idx:
            i = susc[ni]
            inf_date[i] = t
            if np.random.rand() < p_asympto:
                infect_status[i] = 2
                # incubation = np.random.randint(minIncub, maxIncub+1), # note: does not work with STAN
                incubation = np.random.gamma(inc_shape_asymp, inc_scale_asymp)
            else:
                infect_status[i] = 1
                incubation = np.random.gamma(inc_shape_symp, inc_scale_symp)
            obs_time[i] = t + incubation
        t += dt

    # finalize inclusion case
    infected = np.where(infect_status>0)[0]
    if infected.size == 1:
        inc = infected[0]
    else:
        inc = np.random.choice(infected)
    is_inclu[inc] = 1

    # delay for inclusion date
    if infect_status[inc] == 1:
        s, r, sh = delayDist[0], delayDist[1], delayDist[2]
    else:
        s, r, sh = delayDist[3], delayDist[4], delayDist[5]
    delay = np.random.gamma(s, 1.0/r) - sh
    incl_dt[inc] = obs_time[inc] + delay

    # package outputs
    end_time = np.full(hh_size, duration_followup, dtype=np.int64)
    obs_time_final = np.where(np.isnan(obs_time), 1000, obs_time)

    # Re‐anchor dates so that min(date_sympt, inf_date, incl_dt) → t0=30
    all_dates = np.concatenate((obs_time_final, inf_date, incl_dt))
    t0 = np.min(all_dates[all_dates >= 0]) if np.any(all_dates >= 0) else 0
    obs_time_final = np.where(obs_time_final < 1000, 30 + obs_time_final - t0, obs_time_final)
    end_time = 30 + end_time - t0
    inf_date = np.where(inf_date >= 0, 30 + inf_date - t0, -1)
    incl_dt = np.where(incl_dt >= 0, 30 + incl_dt - t0, -1)

    # round dates to nearest integer
    obs_time_final = np.round(obs_time_final).astype(np.int64)
    incl_dt = np.round(incl_dt).astype(np.int64)
    end_time = np.round(end_time).astype(np.int64)

    return {
        'is_index':    is_index,
        'infect_status': infect_status,
        'obs_time':    obs_time_final,
        'end_time':    end_time,
        'incl_dt':     incl_dt,
        'is_inclu':    is_inclu,
        'inf_date':    inf_date,
    }

# -----------------------------------------------------------------------------
# Python wrapper to run across all households
# -----------------------------------------------------------------------------
class OutbreakSimulator:
    def __init__(self, variant='alpha', n_repeat=50):
        self.variant   = variant
        self.n_repeat  = n_repeat
        self.minimal_length = 9  # minimal length of the household PedCov -> time steps

        if variant == "alpha":
            self.p_asympto = 0.22
            mean_incub, sd_incub = 4.42, 2.3
            #self.minIncub, self.maxIncub = 2, 7
            self.shapeIncubAsymp = 4.0  # note: similar to uniform distribution before
            self.scaleIncubAsymp = 1.286

            self.shape_generation_time, self.scale_generation_time = 2, 1/0.44
            self.delayDist = [6.9368753, 0.7376425, -3.0000000, 5.6516107, 0.9719026, 2.0000000]
        elif variant == "omicron":
            self.p_asympto = 0.25
            mean_incub, sd_incub = 3.09, 1.64
            #self.minIncub, self.maxIncub = 1, 5
            self.shapeIncubAsymp = 7.0  # note: similar to uniform distribution before
            self.scaleIncubAsymp = 2/3

            self.shape_generation_time, self.scale_generation_time = 3.531, 1/1.098
            self.delayDist = [8.4310842, 0.9508425, -4.0000000, 3.8209443, 1.2737556, 1.0000000]
        else:
            raise ValueError(f"Unknown variant: {variant}")
        self.shapeIncub = mean_incub**2 / sd_incub**2
        self.scaleIncub = sd_incub**2 / mean_incub


        # info: taken from sophie's code, but not sure why this is needed
        self.delayDist[2] = abs(self.delayDist[2]) + 1 if self.delayDist[2] <= 0 else 0
        self.delayDist[5] = abs(self.delayDist[5]) + 1 if self.delayDist[5] <= 0 else 0

        # load PedCov
        self.df_raw = pd.read_table(f"PedCov/pedcovid_data_structure_{self.variant}.txt", delimiter=' ')

    def __call__(self,
                 alpha, beta, delta,
                 mu_inf_SI, mu_inf_SC, mu_inf_AI, mu_inf_AC, mu_inf_AA,
                 mu_susc_I, mu_susc_C,
                 mu_protect_acq, mu_protect_transm,
                 selection_procedure="random",
                 return_df=False
                 ):
        if isinstance(selection_procedure, str):
            selection_procedure = [selection_procedure]
        for sp in selection_procedure:
            if sp not in ["none", "random", "pedcov", "adultcov"]:
                raise ValueError("Method must be one of: 'none', 'random', 'pedcov', 'adultcov'")

        # Replicate the dataframe n_repeat times
        df = pd.concat([self.df_raw] * self.n_repeat, ignore_index=True)

        # Generate id_rep
        df["id_rep"] = df.groupby(["id_hh", "id_patient"]).cumcount() + 1

        # Recreate the id_patient
        df["id_patient"] = df.groupby("id_hh").cumcount() + 1  # Row number within id_hh

        # Combine id_hh and id_rep
        df["id_hh"] = df["id_hh"].astype(str) + "-" + df["id_rep"].astype(str)

        # Extract the original id_hh
        df["id_hh_origin"] = df["id_hh"].str.extract(r"^([0-9]+)")

        # 2) for each household, run simulate_outbreak
        results = []
        for hh, hh_df in df.groupby("id_hh"):
            hh_size = len(hh_df)
            # extract arrays
            duration  = int(hh_df["duration_followup"].iat[0])
            age_cat   = hh_df["age"].astype(np.int64).values
            protected = hh_df["protected"].astype(np.int64).values

            out = simulate_outbreak(
                hh_size=hh_size,
                duration_followup=duration,
                age_cat=age_cat,
                protected=protected,
                p_asympto=self.p_asympto,
                inc_shape_asymp=self.shapeIncubAsymp,
                inc_scale_asymp=self.scaleIncubAsymp,
                inc_shape_symp=self.shapeIncub,
                inc_scale_symp=self.scaleIncub,
                kt_shape=self.shape_generation_time,
                kt_rate=1/self.scale_generation_time,
                delayDist=self.delayDist,
                alpha=alpha,
                beta=beta,
                delta=delta,
                mu_inf_SI=mu_inf_SI,
                mu_inf_SC=mu_inf_SC,
                mu_inf_AI=mu_inf_AI,
                mu_inf_AC=mu_inf_AC,
                mu_inf_AA=mu_inf_AA,
                mu_susc_I=mu_susc_I,
                mu_susc_C=mu_susc_C,
                mu_protect_acq=mu_protect_acq,
                mu_protect_transm=mu_protect_transm
            )
            # attach outputs back to DataFrame
            hh_df = hh_df.reset_index(drop=True)
            hh_df["inf_date"] = out['inf_date']
            hh_df["date_sympt"] = out['obs_time']  # use obs_time as date_sympt
            hh_df["infect_status"] = out['infect_status']
            hh_df["is_index"] = out['is_index']
            hh_df["is_incluCase"] = out['is_inclu']  # inclusion case
            hh_df["incl_dt"] = out['incl_dt']
            hh_df["end_followup"] = out['end_time']
            results.append(hh_df)

        # 3) combine all households
        new_data_full = pd.concat(results, ignore_index=True)

        # 4) Selection of households
        data_list = []
        norm_data_list = []
        for sp in selection_procedure:
            if sp == 'none':
                new_data = new_data_full.copy()
            else:
                new_data = data_selection(new_data_full, variant=self.variant, method=sp)
            new_data['select_process'] = sp
            new_data['variant'] = self.variant
            new_data = new_data[['id_patient', 'id_hh', 'id_hh_origin', 'hh_size', 'date_sympt',
                                 'infect_status', 'end_followup', 'age', 'age_exact', 'protected',
                                 'variant', 'select_process']]
            # add method to id_hh
            new_data['id_hh'] = new_data['id_hh'] + '-' + sp

            data_list.append(new_data)
            # normalize the household PedCov for input to the neural network
            new_data_norm = normalize_household_data(new_data, minimal_length=self.minimal_length)
            norm_data_list.append(new_data_norm)
        data_list = pd.concat(data_list, ignore_index=True)
        norm_data_list = np.stack(norm_data_list, axis=0)
        if len(selection_procedure) == 1:
            norm_data_list = norm_data_list[0]  # remove first dimension if only one method
        if return_df:
            return dict(sim_data_df=data_list, sim_data=norm_data_list)
        else:
            return dict(sim_data=norm_data_list)

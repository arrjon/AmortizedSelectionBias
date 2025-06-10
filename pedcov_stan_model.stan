/*
  pedcov_coxph.stan
  Household transmission model with latent infection augmentation.

  We augment, per household h:
    D_h ~ Uniform(date_sympt_h, date_sympt_h + 1)      // continuous symptom/test date of index case
    U_h ~ Gamma(... parameters depend on symptom status of index) // incubation period
    τ_h = D_h - (U_h - shift)                         // latent infection time of index

  Transmission hazard for susceptible i at day t:
    λ_i(t) = α
           + β (hh_size[h]/4)^(-δ)
             μ_susc_i
             ∑_{j in H_h, τ_j < t} μ_inf_j μ_transm_j f(t - τ_j)
  where f(lag) = Gamma PDF(kt_shape, kt_scale).
*/

functions {
  // PDF of the generation‐time kernel f(lag)
  real f_lag(real lag, real kt_shape, real kt_scale) {
    if (lag <= 0) return 0;
    // Stan’s gamma_lpdf uses (shape, rate), so rate = 1/scale
    return exp( gamma_lpdf(lag | kt_shape, 1/kt_scale) );
  }
  // CDF of the generation‐time kernel F(lag)
  real F_lag(real lag, real kt_shape, real kt_scale) {
    if (lag <= 0) return 0;
    return gamma_cdf(lag | kt_shape, 1/kt_scale);
  }
}

data {
  int<lower=1>                N;               // number of individuals
  int<lower=1>                H;               // number of households
  int<lower=1>                maxT;            // maximum follow-up day
  array[N] int<lower=1,upper=H>        hh_id;        // household ID for each i
  array[H] int<lower=1>                hh_size;      // size of each household
  array[N] int<lower=0>                end_time;     // censoring or end-of-follow-up day
  array[N] int<lower=0,upper=2>        infect_status;// 0=susceptible,1=symptomatic,2=asymptomatic
  array[N] real<lower=0>               obs_time;        // reported symptom/test day
  array[N] int<lower=0,upper=2>        age_cat;      // 0=infant,1=child,2=adult
  array[N] int<lower=0,upper=1>        is_protected; // unused here, but available

  // Incubation‐period priors
  real<lower=0>               inc_shape_symp;
  real<lower=0>               inc_rate_symp;
  real<lower=0>               inc_shift_symp;
  real<lower=0>               inc_shape_asymp;
  real<lower=0>               inc_rate_asymp;
  real<lower=0>               inc_shift_asymp;

  // Generation‐time kernel parameters
  real<lower=0>               kt_shape;
  real<lower=0>               kt_scale;
}

parameters {
  // 1) Household transmission parameters
  real<lower=0,upper=0.1>     alpha;      // baseline hazard
  real<lower=0,upper=10>      beta;       // transmissibility coefficient
  real<lower=-3,upper=3>      delta;      // exponent on household size

  // 2) Log-multipliers for infectivity by (status × age)
  // real                         log_mu_inf_SC;  // symptomatic child
  // real                         log_mu_inf_SI;  // symptomatic infant/adult
  // real                         log_mu_inf_AI;  // asymptomatic infant
  // real                         log_mu_inf_AC;  // asymptomatic child
  // real                         log_mu_inf_AA;  // asymptomatic adult

  // 3) Log-multipliers for susceptibility by age
  // real                         log_mu_susc_C;  // child
  // real                         log_mu_susc_I;  // infant

  // 4) Protection multiplier
  // real                         log_mu_protect_acq;
  // real                         log_mu_protect_transm;

  // 5) Latent-time offsets for each infected
  vector<lower=0,upper=1>[N]   D;  // Uniform(0,1): symptom/test date offset
  vector<lower=0>[N]           U;  // Gamma: incubation periods
}

transformed parameters {
  vector<lower=0>[N] mu_inf;   // individual infectivity
  vector<lower=0>[N] mu_susc;  // individual susceptibility
  vector[N]          tau;  // latent infection times
  vector<lower=0>[N] mu_protect_acq;   // individual acquisition protection
  vector<lower=0>[N] mu_protect_transm;  // individual transmission protection


  // Map log-multipliers to per-individual parameters
  for (i in 1:N) {
    // Susceptibility by age
    if (age_cat[i] == 1)
      mu_susc[i] = 1; // exp(log_mu_susc_C);
    else
      mu_susc[i] = 1; // exp(log_mu_susc_I);

    // Infectivity by status and age
    if (infect_status[i] == 1) {
      // symptomatic
      if (age_cat[i] == 1)          mu_inf[i] = 1; // exp(log_mu_inf_SC);
      else                           mu_inf[i] = 1; // exp(log_mu_inf_SI);
    }
    else if (infect_status[i] == 2) {
      // asymptomatic
      if      (age_cat[i] == 0)     mu_inf[i] = 1; // exp(log_mu_inf_AI);
      else if (age_cat[i] == 1)     mu_inf[i] = 1; // exp(log_mu_inf_AC);
      else                           mu_inf[i] = 1; // exp(log_mu_inf_AA);
    }
    else {
      // never infected → zero infectivity
      mu_inf[i] = 0;
    }

    // Latent infection time only defined for infected
    if (infect_status[i] == 1)
      tau[i] = obs_time[i] + D[i] - (U[i] - inc_shift_symp); // shift for symptomatic
    else if (infect_status[i] == 2)
      tau[i] = obs_time[i] + D[i] - (U[i] - inc_shift_asymp); // shift for asymptomatic
    else
      tau[i] = 0;  // unused for susceptibles

    // Protection multipliers
    if (is_protected[i] == 1) {
      mu_protect_acq[i] = 0.8; // exp(log_mu_protect_acq);
      mu_protect_transm[i] = 1.0; // exp(log_mu_protect_transm);
    } else {
      // not protected
      mu_protect_acq[i] = 1.0;
      mu_protect_transm[i] = 1.0;
    }
  }
}

model {
  // 1) Priors on transmission parameters
  alpha           ~ uniform(0, 0.1);
  beta            ~ uniform(0, 2);
  delta           ~ uniform(-3, 3);

  // 2) Priors on log-multipliers
  // log_mu_inf_SC   ~ normal(0, 0.7);
  // log_mu_inf_SI   ~ normal(0, 0.7);
  // log_mu_inf_AI   ~ normal(0, 0.7);
  // log_mu_inf_AC   ~ normal(0, 0.7);
  // log_mu_inf_AA   ~ normal(0, 0.7);

  // log_mu_susc_C   ~ normal(0, 0.7);
  // log_mu_susc_I   ~ normal(0, 0.7);

  // log_mu_protect_acq   ~ normal(0, 0.7);
  // log_mu_protect_transm   ~ normal(0, 0.7);

  // 3) Priors for latent-time offsets
  D               ~ uniform(0, 1);
  for (i in 1:N) {
    if (infect_status[i] == 1)
      U[i] ~ gamma(inc_shape_symp, inc_rate_symp);
    else if (infect_status[i] == 2)
      U[i] ~ gamma(inc_shape_asymp, inc_rate_asymp);
    else
      U[i] ~ gamma(1, 1);  // dummy prior for susceptibles
  }

  // 4) Transmission‐hazard likelihood
  for (i in 1:N) {
    // household‐size weighting
    real w = pow(hh_size[hh_id[i]] / 4.0, -delta);

    if (infect_status[i] > 0) {
      // — Event at τ_i for infected i
      real sum_haz = 0;
      real sum_cum = 0;
      for (j in 1:N) {
        if (infect_status[j] > 0
            && hh_id[j] == hh_id[i]
            && tau[j] < tau[i]) {
          real lag = tau[i] - tau[j];
          sum_haz += mu_protect_transm[j] * mu_inf[j] * f_lag(lag, kt_shape, kt_scale);
          sum_cum += mu_protect_transm[j] * mu_inf[j] * F_lag(lag, kt_shape, kt_scale);
        }
      }
      real lambda_i = alpha + beta * w * mu_protect_acq[i] * mu_susc[i] * sum_haz;
      real H_i      = alpha * tau[i]   + beta * w * mu_protect_acq[i] * mu_susc[i] * sum_cum;
      target += log(lambda_i) - H_i;

    } else {
      // — Survival term for susceptibles up to end_time[i]
      real T       = end_time[i];
      real sum_cum = 0;
      for (j in 1:N) {
        if (infect_status[j] > 0
            && hh_id[j] == hh_id[i]
            && tau[j] < T) {
          real lag = T - tau[j];
          sum_cum += mu_protect_transm[j] * mu_inf[j] * F_lag(lag, kt_shape, kt_scale);
        }
      }
      real H_i = alpha * T + beta * w * mu_protect_acq[i] * mu_susc[i] * sum_cum;
      target += -H_i;
    }
  }
}

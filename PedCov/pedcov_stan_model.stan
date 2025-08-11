/*
  pedcov_stan_model.stan
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
  where f(lag) = Gamma PDF(kt_shape, kt_rate).
*/

functions {
  // PDF of the generation‐time kernel f(lag)
  real f_lag(real lag, real kt_shape, real kt_rate) {
    if (lag <= 0) return 0;
    // Stan’s gamma_lpdf uses (shape, rate), so rate = 1/scale
    return exp( gamma_lpdf(lag | kt_shape, kt_rate) );
  }
  // CDF of the generation‐time kernel F(lag)
  real F_lag(real lag, real kt_shape, real kt_rate) {
    if (lag <= 0) return 0;
    return gamma_cdf(lag | kt_shape, kt_rate);
  }
}

data {
  int<lower=1>                         N;            // number of individuals
  int<lower=1>                         H;            // number of households
  array[N] int<lower=1,upper=H>        hh_id;        // household ID for each i
  array[H] int<lower=1>                hh_size;      // size of each household
  array[N] int<lower=0>                end_time;     // censoring or end-of-follow-up day
  array[N] int<lower=0,upper=2>        infect_status;// 0=susceptible,1=symptomatic,2=asymptomatic
  array[N] real<lower=0>               obs_time;     // reported symptom/test day
  array[N] int<lower=0,upper=2>        age_cat;      // 0=infant,1=child,2=adult
  array[N] int<lower=0,upper=1>        is_protected; // unused here, but available
  array[N] int                         last_test_neg;     // date of last negative test (-1000 if no information)
  array[N] int<lower=0>                first_test_pos;     // first positive test for symptomatic, and the LAST positive test for asymptomatic

  // Helper indices
  int<lower=0>                               n_infected;
  int<lower=0>                               n_susceptible;
  array[n_infected] int<lower=1,upper=N>     infected_idx;
  array[n_susceptible] int<lower=1,upper=N>  susceptible_idx;
  array[H] int<lower=1,upper=N>              hh_start_idx;
  array[H] int<lower=1,upper=N>              hh_end_idx;
  array[H] int<lower=0>              first_infected_idx;


  // Incubation‐period priors
  real<lower=0>               inc_shape_symp;
  real<lower=0>               inc_rate_symp;
  real<lower=0>               penalty_strength;  // strength of penalty for latent infection time outside bounds

  // Generation‐time kernel parameters
  real<lower=0>               kt_shape;
  real<lower=0>               kt_rate;

  // fixed params
  // real<lower=0>               mu_protect_acq;
  // real<lower=0>               mu_protect_transm;
}


transformed data {
  vector<lower=0>[N] u_min;  // minimum latent infection time
  vector<lower=0>[N] u_max;  // maximum latent infection time

  for (i in 1:N) {
    if (infect_status[i] == 1) {
      // Infected: infection must occur between 1 day before last neg test and 1 day before first pos test
      u_min[i] = obs_time[i] - (first_test_pos[i] - 1);
      u_max[i] = obs_time[i] - (last_test_neg[i] - 1);
    } else if (infect_status[i] == 2) {
      // Unconfirmed infection: obs_time is first positive test date
      u_min[i] = 0;

      if (first_test_pos[i] > 0 && (first_test_pos[i] - obs_time[i]) > 15) {
        // No constraint - time span too large to be reliable
        u_max[i] = positive_infinity();
      } else {

          u_max[i] = 15; // maximum incubation period
          if (last_test_neg[i] > 0) {
              // If last negative test was at day X, infection was before day X
              u_max[i] = fmin(u_max[i], obs_time[i] - (last_test_neg[i] - 1));
          }

          if (first_test_pos[i] > 0) {
              // If first positive test was at day Y, infection was before day Y
              u_min[i] = fmax(u_min[i], obs_time[i] - (first_test_pos[i] - 1));
          }
      }
    } else {
      // Susceptible or uninfected: bounds unused
      u_min[i] = 0;
      u_max[i] = 0;
    }

    // Ensure bounds are valid
    if (u_min[i] < 0) u_min[i] = 0;
    if (u_max[i] < u_min[i]) {  // no valid range
      u_max[i] = obs_time[i];
      u_min[i] = 0;
    }
  }
}

parameters {
  // 1) Household transmission parameters
  real<lower=0,upper=0.1>       alpha;      // baseline hazard
  real<lower=0>                 beta;       // transmissibility coefficient
  real                          delta;      // exponent on household size

  // 2) Log-multipliers for infectivity by (status × age)
  real<lower=0>                mu_inf_SI;  // symptomatic infant/adult
  real<lower=0>                mu_inf_SC;  // symptomatic child
  real<lower=0>                mu_inf_AI;  // asymptomatic infant
  real<lower=0>                mu_inf_AC;  // asymptomatic child
  real<lower=0>                mu_inf_AA;  // asymptomatic adult

  // 3) Log-multipliers for susceptibility by age
  real<lower=0>                mu_susc_I;  // infant
  real<lower=0>                mu_susc_C;  // child

  // 4) Protection multiplier
  real<lower=0>                mu_protect_acq;
  real<lower=0>                mu_protect_transm;

  // 5) Latent-time offsets for each infected
  vector<lower=0,upper=30>[N]           U;  // Gamma: incubation periods
}

transformed parameters {
  vector<lower=0>[N] mu_inf_array;             // individual infectivity
  vector<lower=0>[N] mu_susc_array;            // individual susceptibility
  vector[N]          tau;                      // latent infection times
  vector<lower=0>[N] mu_protect_acq_array;     // individual acquisition protection
  vector<lower=0>[N] mu_protect_transm_array;  // individual transmission protection
  vector[N]          w;                        // household size weights

  // Map log-multipliers to per-individual parameters
  for (i in 1:N) {
    // Susceptibility by age
    if      (age_cat[i] == 0)  mu_susc_array[i] = mu_susc_I;
    else if (age_cat[i] == 1)  mu_susc_array[i] = mu_susc_C;
    else                       mu_susc_array[i] = 1.0;  // adults have no susceptibility multiplier (reference group)

    // Infectivity by status and age
    if (infect_status[i] == 1) {
      // symptomatic
      if      (age_cat[i] == 0)  mu_inf_array[i] = mu_inf_SI;
      else if (age_cat[i] == 1)  mu_inf_array[i] = mu_inf_SC;
      else                       mu_inf_array[i] = 1.0;  // adults have no infectivity multiplier (reference group)
    }
    else if (infect_status[i] == 2) {
      // asymptomatic
      if      (age_cat[i] == 0)     mu_inf_array[i] = mu_inf_AI;
      else if (age_cat[i] == 1)     mu_inf_array[i] = mu_inf_AC;
      else                          mu_inf_array[i] = mu_inf_AA;
    }
    else {
      // never infected, should never contribute to hazard
      mu_inf_array[i] = 0;
    }

    // Latent infection time only defined for infected
    if (infect_status[i] == 1)
      tau[i] = obs_time[i] - U[i];
    else if (infect_status[i] == 2)
      tau[i] = obs_time[i] - U[i];
    else
      tau[i] = 0;  // unused for susceptibles

    // Protection multipliers
    if (is_protected[i] == 1) {
      mu_protect_acq_array[i] = mu_protect_acq;
      mu_protect_transm_array[i] = mu_protect_transm;
    } else {
      // not protected
      mu_protect_acq_array[i] = 1.0;
      mu_protect_transm_array[i] = 1.0;
    }

    // Precompute household weights
    w[i] = pow(hh_size[hh_id[i]] / 4.0, -delta);
  }
}

model {
  // 1) Priors on transmission parameters
  alpha                 ~ uniform(0.0, 0.1);
  beta                  ~ gamma(2.0, 2.0);
  delta                 ~ normal(0.0, 1.0);

  // 2) Priors on log-multipliers
  mu_inf_SC             ~ lognormal(0, 0.7);
  mu_inf_SI             ~ lognormal(0, 0.7);
  mu_inf_AI             ~ lognormal(0, 0.7);
  mu_inf_AC             ~ lognormal(0, 0.7);
  mu_inf_AA             ~ lognormal(0, 0.7);

  mu_susc_C             ~ lognormal(0, 0.7);
  mu_susc_I             ~ lognormal(0, 0.7);

  mu_protect_acq        ~ lognormal(0, 0.7);
  mu_protect_transm     ~ lognormal(0, 0.7);

  // 4) Transmission‐hazard likelihood
  // Process infected individuals
  for (inf_idx in 1:n_infected) {
    int i = infected_idx[inf_idx];
    int household = hh_id[i];
    real first_tau = tau[hh_start_idx[household]+first_infected_idx[household]];

    // Priors for latent-time offsets
    //if (infect_status[i] == 1)
     // U[i] ~ gamma(inc_shape_symp, inc_rate_symp);
    //else // if (infect_status[i] == 2)
      // U[i] ~ gamma(inc_shape_asymp, inc_rate_asymp);
    // Gamma prior on U
    if (infect_status[i] == 1)
        target += gamma_lpdf(U[i] | inc_shape_symp, inc_rate_symp);

    // Soft penalty for being outside of bounds
    if (U[i] < u_min[i]) {
      target += -penalty_strength * square(U[i] - u_min[i]);
    } else if (U[i] > u_max[i]) {
      target += -penalty_strength * square(U[i] - u_max[i]);
    }

    // — Event at τ_i for infected i
    real sum_haz = 0;
    real sum_cum = 0;

    // Only loop over household members (precomputed ranges)
    // tau[j_idx] < tau[i] excludes the first infected in the household
    for (j_idx in hh_start_idx[household]:hh_end_idx[household]) {
      // if (infect_status[j_idx] > 0 && tau[j_idx] < tau[i]) { -> mu_inf_array is 0 for susceptibles,
        real lag = tau[i] - tau[j_idx];
        real contrib = mu_protect_transm_array[j_idx] * mu_inf_array[j_idx];
        sum_haz += contrib * f_lag(lag, kt_shape, kt_rate);  // returns 0 for lag <= 0
        sum_cum += contrib * F_lag(lag, kt_shape, kt_rate);
      // }
    }

    real lambda_i = alpha + beta * w[i] * mu_protect_acq_array[i] * mu_susc_array[i] * sum_haz;
    real H_i = alpha * (tau[i]-first_tau) + beta * w[i] * mu_protect_acq_array[i] * mu_susc_array[i] * sum_cum;
    target += log(lambda_i) - H_i;
  }

  // Process susceptible individuals
  for (sus_idx in 1:n_susceptible) {
    // — Survival term for susceptibles up to end_time[i]
    int i = susceptible_idx[sus_idx];
    int household = hh_id[i];
    real T = end_time[i];
    real first_tau = tau[hh_start_idx[household]+first_infected_idx[household]];

    // Priors for latent-time offsets
    U[i] ~ gamma(1, 1);  // dummy prior for susceptibles

    real sum_cum = 0;

    // Only loop over household members
    for (j_idx in hh_start_idx[household]:hh_end_idx[household]) {
      // if (infect_status[j_idx] > 0 && tau[j_idx] < T) { -> mu_inf_array is 0 for susceptibles,
        real lag = T - tau[j_idx];
        sum_cum += mu_protect_transm_array[j_idx] * mu_inf_array[j_idx] * F_lag(lag, kt_shape, kt_rate);  // returns 0 for lag <= 0
      // }
    }

    real H_i = alpha * (T-first_tau) + beta * w[i] * mu_protect_acq_array[i] * mu_susc_array[i] * sum_cum;
    target += -H_i;
  }
}

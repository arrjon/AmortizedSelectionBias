functions {
  //---------------------------------------------------------------------------
  // 1) Compute discretized generation-time PDF up to maxT
  //---------------------------------------------------------------------------
  vector gen_time_pdf(int maxT,
                      real shapeInf,
                      real scaleInf) {
    vector[maxT+1] out;
    // Continuous gamma CDF at integer points
    for (t in 0:maxT) {
      if (t == 0) {
        out[t+1] = 0;               // no mass at t<1
      } else {
        real F_t   = gamma_cdf(t | shapeInf, scaleInf);
        real F_tm1 = gamma_cdf(t-1 | shapeInf, scaleInf);
        out[t+1]   = F_t - F_tm1;   // P(T ∈ (t-1, t]]
      }
    }
    return out;
  }

  //---------------------------------------------------------------------------
  // 2) Single-time, single-individual force-of-infection
  //---------------------------------------------------------------------------
  real compute_foi(int       h_id,
                   int       i,
                   array[] int hh_members,
                   array[] int inf_time,       // inf_time[j] = infection time of member j (0 if never)
                   array[] int infect_status,  // 0=susceptible,1=sympto,2=asympto
                   int       t,
                   int       hh_size,
                   array[] int age_cat,        // 0=infant,1=child,2=adult
                   array[] int is_protected,      // 0/1
                   real      mu_inf_SC,
                   real      mu_inf_SI,
                   real      mu_inf_AI,
                   real      mu_inf_AC,
                   real      mu_inf_AA,
                   real      mu_susc_C,
                   real      mu_susc_I,
                   real      mu_protect_acq,
                   real      mu_protect_transm,
                   real      alpha,
                   real      beta,
                   real      delta,
                   vector    gtime_pdf) {
    real beta_foyer = 0;
    // sum over all infectious j≠i
    for (idx in 1:hh_size) {
      int j = hh_members[idx];
      if (j == i) continue;
      int tau = t - inf_time[j];
      if (inf_time[j] > 0 && tau >= 1 && tau <= size(gtime_pdf)-1) {
        real base = gtime_pdf[tau+1] / (1 - gtime_pdf[1]);
        // skip impossible infinities
        if (!is_inf(base)) {
          // relative infectiousness by age/status
          int as = infect_status[j];
          int ac = age_cat[j];
          real mu_inf_val;
          // map (ac,as) → mu_inf
          if      (ac == 1 && as == 1) mu_inf_val = mu_inf_SC;  // SC
          else if (ac == 0 && as == 1) mu_inf_val = mu_inf_SI;  // SI
          else if (ac == 0 && as == 2) mu_inf_val = mu_inf_AI;  // AI
          else if (ac == 1 && as == 2) mu_inf_val = mu_inf_AC;  // AC
          else                         mu_inf_val = mu_inf_AA;  // AA
          real contrib = base * mu_inf_val;
          // protection on transmission
          if (is_protected[j] == 1) contrib *= mu_protect_transm;
          beta_foyer += contrib;
        }
      }
    }
    // susceptibility of i by age
    if      (age_cat[i] == 1)  beta_foyer *= mu_susc_C;
    else if (age_cat[i] == 0)  beta_foyer *= mu_susc_I;
    // protection on acquisition
    if (is_protected[i] == 1) beta_foyer *= mu_protect_acq;
    // final additive hazard
    return alpha + beta / pow(hh_size/4.0, delta) * beta_foyer;
    // h_t = fmin(h_t, 1.0);  // cap at 1
    // h_t = inv_logit(h_t); // Logistic function to constrain h_t between 0 and 1
    // return h_t;
  }
}

data {
  int<lower=1> N;                   // total individuals
  int<lower=1> H;                   // total households
  int<lower=1> maxT;                // maximum follow-up time
  array[N] int<lower=1,upper=H> hh_id;    // household index for each individual
  array[H] int<lower=1> hh_size;          // size of each hh
  array[N] real<lower=0,upper=maxT> obs_time;        // censoring / follow-up end
  array[N] int<lower=0,upper=2> infect_status;      // 0/1/2
  array[N] int<lower=0> inf_time_obs;        // observed infection time (0 if never)
  array[N] int<lower=0,upper=2> age_cat;             // 0=infant,1=child,2=adult
  array[N] int<lower=0,upper=1> is_protected;           // 0/1
  real shapeInf;   // Gamma shape parameter for generation time
  real scaleInf;  // Gamma distribution parameters for generation time
}

transformed data {
  // Preallocate per-household member lists
  array[H, max(hh_size)] int hh_members;   // padded with zeros

  // Initialize with zeros
  for (h in 1:H) {
    for (j in 1:max(hh_size)) {
      hh_members[h, j] = 0;
    }
  }

  // Fill hh_members with indices
  {
    array[H] int counts = rep_array(0, H);
    for (i in 1:N) {
      int h = hh_id[i];
      counts[h] += 1;
      hh_members[h, counts[h]] = i;
    }
  }

  // Precompute generation-time PDF
  vector[maxT+1] g_pdf = gen_time_pdf(maxT, shapeInf, scaleInf);
}

parameters {
  real log_alpha;
  // real log_beta;
  // real log_delta;

  // real log_mu_inf_SC;     // Symptomatic Child
  // real log_mu_inf_SI;     // Symptomatic Infant
  // real log_mu_inf_AI;     // Asymptomatic Infant
  // real log_mu_inf_AC;     // Asymptomatic Child
  // real log_mu_inf_AA;     // Asymptomatic Adult

  // real log_mu_susc_C;     // Child susceptibility
  // real log_mu_susc_I;     // Infant susceptibility

  // real log_mu_protect_acq;    // Protection acquisition
  // real log_mu_protect_transm; // Protection transmission
}

transformed parameters {
  real<lower=0> alpha = 0.001; //exp(log_alpha);
  real<lower=0> beta = 0; //exp(log_beta);
  real<lower=0> delta = 1.3; //exp(log_delta);

  real<lower=0> mu_inf_SC = 1.0; //exp(log_mu_inf_SC);     // Symptomatic Child
  real<lower=0> mu_inf_SI = 1.0; //exp(log_mu_inf_SI);     // Symptomatic Infant
  real<lower=0> mu_inf_AI = 1.0; //exp(log_mu_inf_AI);     // Asymptomatic Infant
  real<lower=0> mu_inf_AC = 1.0; //exp(log_mu_inf_AC);     // Asymptomatic Child
  real<lower=0> mu_inf_AA = 1.0; //exp(log_mu_inf_AA);     // Asymptomatic Adult

  real<lower=0> mu_susc_C = 1.0; // exp(log_mu_susc_C);     // Child susceptibility
  real<lower=0> mu_susc_I = 1.0; //exp(log_mu_susc_I);     // Infant susceptibility

  //real<lower=0> mu_protect_acq = exp(log_mu_protect_acq);    // Protection acquisition
  //real<lower=0> mu_protect_transm = exp(log_mu_protect_transm); // Protection transmission
  real<lower=0> mu_protect_acq = 0.8;
  real<lower=0> mu_protect_transm = 1.0;
}

model {
  // Priors
  log_alpha      ~ normal(-10, 0.7);
  // log_beta       ~ normal(0, 0.7);
  // log_delta      ~ normal(0, 0.7);

  // log_mu_inf_SC     ~ normal(0, 0.7);
  // log_mu_inf_SI     ~ normal(0, 0.7);
  // log_mu_inf_AI     ~ normal(0, 0.7);
  // log_mu_inf_AC     ~ normal(0, 0.7);
  // log_mu_inf_AA     ~ normal(0, 0.7);

  // log_mu_susc_C     ~ normal(0, 0.7);
  // log_mu_susc_I     ~ normal(0, 0.7);

  //log_mu_protect_acq    ~ normal(0, 0.7);
  //log_mu_protect_transm ~ normal(0, 0.7);

  // Discrete-time survival/hazard likelihood
  for (h in 1:H) {
    int sz = hh_size[h];
    for (idx in 1:sz) {
      int i = hh_members[h, idx];
      if (i > 0) {  // check for valid member (not padding)
        // loop over time 1…obs_time[i]
        for (t in 1:to_int(obs_time[i])) {
          real h_t = compute_foi(
            h, i, hh_members[h],
            inf_time_obs, infect_status,
            t, sz,
            age_cat, is_protected,
            mu_inf_SC, mu_inf_SI, mu_inf_AI, mu_inf_AC, mu_inf_AA,
            mu_susc_C, mu_susc_I,
            mu_protect_acq, mu_protect_transm,
            alpha, beta, delta,
            g_pdf
          );
          real p_t = 1 - exp(-h_t);
          if (inf_time_obs[i] == t) {
            // infection event at time t
            target += bernoulli_lpmf(1 | p_t);
          } else if (inf_time_obs[i] == 0 || t < inf_time_obs[i]) {
            // survived time t (never infected OR infected later)
            target += bernoulli_lpmf(0 | p_t);
          }
        }
      }
    }
  }
}

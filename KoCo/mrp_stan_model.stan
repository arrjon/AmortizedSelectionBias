// same assumptions as the NPE, except the missingness handling (complete cases only).
data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> y;
  matrix[N, 10] X;                  // dummies in make_priors() coefficient order
  int<lower=1> C;
  matrix[C, 10] X_cells;            // poststratification cells
  vector<lower=0>[C] N_c;           // cell weights
  real<lower=0, upper=1> sens;
  real<lower=0, upper=1> spec;
}
parameters {
  real alpha;
  vector[10] beta;
}
model {
  vector[N] p = inv_logit(alpha + X * beta);
  alpha ~ normal(-3, 1);
  beta ~ normal(0, 0.5);
  y ~ bernoulli(sens * p + (1 - spec) * (1 - p));
}
generated quantities {
  real prevalence = dot_product(N_c, inv_logit(alpha + X_cells * beta)) / sum(N_c);
}

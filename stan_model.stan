data {
      int<lower=0> N;                    // Number of observations
      array[N] int<lower=0, upper=1> T;  // Treatment indicator (0 or 1)
      vector[N] Y;                       // Observed outcomes
}

parameters {
      real theta;                        // Treatment effect parameter
      real<lower=0> sigma;               // Standard deviation of latent z
      vector[N] z;                       // Latent confounder for each observation
    }

model {
      // Priors
      theta ~ uniform(0, 3);
      sigma ~ uniform(0, 3);

      // Prior for latent z
      z ~ normal(0, sigma);

      // Selection model: Y is observed with probability p(1/(1+e^(-z)))
      // This is implicitly handled by only modeling observed data points
      // and accounting for their selection probability in the likelihood

      // Likelihood for observed Y_i
      for (i in 1:N) {
        // Selection probability
        target += log(inv_logit(z[i]));  // p(selection|z) = 1/(1+e^(-z))

        // Data model: Y_i = theta * T_i + z_i + epsilon_i, where epsilon_i ~ N(0, 0.5²)
        Y[i] ~ normal(theta * T[i] + z[i], 0.5);
      }
}

# illness_death_simulation.R
# Clean script for illness-death model simulation
# Can be called from Python using rpy2 or reticulate

####################
# Package loading
####################
packages <- c("survival", "mvna", "parallel", "doMC")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

####################
# illnessdeath_weibull: Simulate illness-death data with weibull baselines
####################
illnessdeath_weibull = function(n.indiv,
                                a01, a02, a12,
                                shape01, shape02, shape12,
                                beta01.sex, beta02.sex, beta12.sex,
                                beta01.age, beta02.age, beta12.age,
                                cov_sex, cov_age, age.center.scale = TRUE,
                                min.time = 1
                                ) {
  age = cov_age
  if (age.center.scale) {
    age = as.numeric(scale(cov_age))  # standardize age
  }
  sex = cov_sex

  # Individual linear predictors for 0→1 (illness) and 0→2 (direct death)
  eta01 = beta01.sex * sex + beta01.age * age
  eta02 = beta02.sex * sex + beta02.age * age
  n.indiv = length(sex)

  # Draw cause specific times from Weibull models
  # Baseline hazard for transition j: h0j(t) = shapej * aj * t^(shapej - 1)
  # Cumulative hazard: Hj(t) = aj * t^shapej
  # Inversion sampler: Tj = { log(1 / U) / (aj * exp(etaj)) }^(1 / shapej)

  U01 = runif(n.indiv)
  U02 = runif(n.indiv)

  t01 = pmax((log(1 / U01) / (a01 * exp(eta01)))^(1 / shape01), min.time)
  t02 = pmax((log(1 / U02) / (a02 * exp(eta02)))^(1 / shape02), min.time)

  # Time to first event and type of first event
  t_first = pmin(t01, t02)
  ill_first = as.integer(t01 <= t02)  # 1 means illness first, 0 means direct death

  # Illness indicators and times
  illt = t_first
  ills = ill_first

  # Initialize death variables
  dt = rep(NA_real_, n.indiv)
  ds = rep(0L, n.indiv)

  # Those with illness first: simulate time from illness to death (1→2 transition)
  idx_ill = which(ills == 1)
  if (length(idx_ill) > 0) {
    eta12 = beta12.sex * sex[idx_ill] + beta12.age * age[idx_ill]

    U12 = runif(length(idx_ill))
    t12 = pmax((log(1 / U12) / (a12 * exp(eta12)))^(1 / shape12), min.time)

    dt[idx_ill] = illt[idx_ill] + t12
    ds[idx_ill] = 1L
  }

  # Those with direct death from state 0
  idx_direct = which(ills == 0)
  if (length(idx_direct) > 0) {
    dt[idx_direct] = t_first[idx_direct]
    ds[idx_direct] = 1L
  }

  data.frame(
    id = seq_len(n.indiv),
    illt = illt,
    ills = ills,
    dt = dt,
    ds = ds,
    sex = sex,
    age = age,
    age_raw = cov_age
  )
}

####################
# truncateData: Apply interval censoring for semi-competing risks
####################
truncateData <- function(data, obs_times, illt = "illt", dt = "dt", max.time = 1825) {
  # obs_times: matrix with n.indiv rows and 2 columns (visit1, visit2)

  n.indiv <- nrow(data)
  censored_data <- data

  # Add observation times
  censored_data$visit1 <- obs_times[, 1]
  censored_data$visit2 <- obs_times[, 2]

  # Validate inputs
  if (any(censored_data$visit1 >= censored_data$visit2)) {
    stop("visit1 must be strictly less than visit2 for all individuals")
  }

  for (i in 1:n.indiv) {
    visit1 <- censored_data$visit1[i]
    visit2 <- censored_data$visit2[i]
    illness_time <- censored_data[[illt]][i]
    death_time <- censored_data[[dt]][i]
    illness_status <- censored_data$ills[i]
    death_status <- censored_data$ds[i]

    # Apply administrative censoring for death
    if (death_time > max.time) {
      censored_data$ds[i] <- 0
      censored_data[[dt]][i] <- max.time
      death_time <- max.time
      death_status <- 0
    }
    # If illness also beyond study end, censor it
    if (illness_time > max.time) {
      censored_data$ills[i] <- 0
      censored_data[[illt]][i] <- max.time
      illness_time <- max.time
      illness_status <- 0
    }

    # Apply interval censoring for illness (only observable at visits)
    censored_data <- censor_illness(censored_data, i, visit1, visit2,
                                    illness_time, death_time,
                                    illness_status, death_status,
                                    illt, max.time)
  }

  list(CensVisit = censored_data)
}

####################
# Helper function: Apply illness interval censoring based on event pattern
####################
censor_illness <- function(data, i, visit1, visit2, illness_time, death_time,
                           illness_status, death_status, illt, max.time) {

  # Case 1: No illness observed, no death
  if (illness_status == 0 && death_status == 0) {
    if (runif(1) < 0.05) {
      # Patient drops out of study after study end, not observed to be ill
      data[[illt]][i] <- visit2
    }
    else {  # Patient is disease-free until end of study and checked again in the next round
      data[[illt]][i] <- max.time
    }
  }

  # Case 2: Illness observed, no death
  else if (illness_status == 1 && death_status == 0) {
    # Snap illness time to the visit where it was detected
    if (illness_time <= visit1) {
      data[[illt]][i] <- visit1
    } else if (illness_time <= visit2) {
      data[[illt]][i] <- visit2
    } else {
      # Illness after last visit: censor to visit2
      data$ills[i] <- 0
      data[[illt]][i] <- visit2
    }
  }

  # Case 3: Illness observed, death observed
  else if (illness_status == 1 && death_status == 1) {

    if (illness_time <= visit1 && death_time > visit1) {
      # Illness detected at/before visit1, death after visit1
      data[[illt]][i] <- visit1

    } else if (illness_time > visit1 && illness_time <= visit2 && death_time > visit2) {
      # Illness between visits, detected at visit2, death after visit2
      data[[illt]][i] <- visit2

    } else if (illness_time > visit1 && death_time <= visit2) {
      # Both events between visit1 and visit2: illness not observable
      data$ills[i] <- 0
      data[[illt]][i] <- visit1

    } else {
      # Illness after visit2: censor to visit2
      data$ills[i] <- 0
      data[[illt]][i] <- visit2
    }
  }

  # Case 4: No illness observed, death observed
  else if (illness_status == 0 && death_status == 1) {

    if (death_time > visit1 && death_time < visit2) {
      # Death after visit1: patient was disease-free at visit1
      data[[illt]][i] <- visit1

    } else if (death_time > visit2) {
      # Death after visit2: patient was disease-free at visit2
      data[[illt]][i] <- visit2

    } else {
      # Death before visit1: patient was disease-free at time 0
      data[[illt]][i] <- 0
    }
  }

  return(data)
}

####################
# datagen: Main simulation function
####################
datagen <- function(i,
                    n.indiv,
                    a01, a02, a12,
                    shape01, shape02, shape12,
                    beta01.sex, beta02.sex, beta12.sex,
                    beta01.age, beta02.age, beta12.age,
                    sex, age_raw, age.center.scale,
                    obs_times, max.time) {
  # Generate full data
  fulldata <- illnessdeath_weibull(
    a01 = a01, a02 = a02, a12 = a12,
    shape01 = shape01, shape02 = shape02, shape12 = shape12,
    beta01.sex = beta01.sex, beta02.sex = beta02.sex, beta12.sex = beta12.sex,
    beta01.age = beta01.age, beta02.age = beta02.age, beta12.age = beta12.age,
    cov_sex = sex, cov_age = age_raw, age.center.scale = age.center.scale
  )

  # Apply censoring schemes
  missdata <- truncateData(fulldata, obs_times, illt = "illt", dt = "dt", max.time = max.time)

  # float to integer
  fulldata$illt <- round(fulldata$illt)
  fulldata$dt <- round(fulldata$dt)
  missdata$CensVisit$illt <- round(missdata$CensVisit$illt)
  missdata$CensVisit$dt <- round(missdata$CensVisit$dt)
  
  list(
    fulldata = fulldata,
    CensVisit = missdata$CensVisit
  )
}


####################
# simulate_from_priors_df: Main interface function for Python
####################
simulate_from_priors_df <- function(priors = NA,
                                    params = NA,
                                    N = 1,
                                    age.center.scale = TRUE,
                                    obs.time = c(800, 1800),
                                    max.time = 1825,
                                    ncores = 1) {

  if (ncores > 1) {
    registerDoMC(cores = ncores)
  }

  draw_hazards <- function(p) {
    c(
      a01 = rgamma(1, shape = p$a01$alpha, rate = p$a01$beta),
      a02 = rgamma(1, shape = p$a02$alpha, rate = p$a02$beta),
      a12 = rgamma(1, shape = p$a12$alpha, rate = p$a12$beta),
      shape01 = rgamma(1, shape = p$shape01$alpha, rate = p$shape01$beta),
      shape02 = rgamma(1, shape = p$shape02$alpha, rate = p$shape02$beta),
      shape12 = rgamma(1, shape = p$shape12$alpha, rate = p$shape12$beta),
      beta01.sex = rnorm(1, mean = p$beta01_sex$mean, sd = p$beta01_sex$sd),
      beta02.sex = rnorm(1, mean = p$beta02_sex$mean, sd = p$beta02_sex$sd),
      beta12.sex = rnorm(1, mean = p$beta12_sex$mean, sd = p$beta12_sex$sd),
      beta01.age = rnorm(1, mean = p$beta01_age$mean, sd = p$beta01_age$sd),
      beta02.age = rnorm(1, mean = p$beta02_age$mean, sd = p$beta02_age$sd),
      beta12.age = rnorm(1, mean = p$beta12_age$mean, sd = p$beta12_age$sd)
    )
  }

  all_results <- list()

  for (ep in names(priors)) {
    # Covariates sex (binary) and age (continuous)
    df <- read.csv(paste0("data/", ep, "_CV.csv"))
    sex = df$sex
    age_raw = df$age
    n.indiv = length(df$age)

    # Visit 1: early observations
    df_1 <- df[df$illt > 0 & df$illt < 1000, ]
    visit1_typical <- median(df_1$illt)

    # Visit 2: late observations
    df_2 <- df[df$illt > 1000 & df$illt < max.time, ]
    visit2_typical <- median(df_2$illt)

    for (i in 1:N) {
      if (is.na(params)) {
         haz <- draw_hazards(priors[[ep]])
      } else {
         haz <- unlist(params[[ep]], use.names = TRUE)
      }

      visit1 <- visit1_typical + rnorm(n.indiv, 0, 100)
      visit2 <- visit2_typical + rnorm(n.indiv, 0, 150)
      v1 <- pmin(visit1, visit2)
      v2 <- pmax(visit1, visit2)
      obs.time <- cbind(pmax(0, pmin(v1, max.time)), pmax(0, pmin(v2, max.time)))

      if (is.vector(obs.time) && length(obs.time) == 2) {
        obs_times <- matrix(rep(obs.time, each = n.indiv), ncol = 2, byrow = TRUE)
      } else if (is.matrix(obs.time)) {
        if (nrow(obs.time) != n.indiv || ncol(obs.time) != 2) {
          stop("obs.time matrix must have n.indiv rows and 2 columns")
        }
        obs_times <- obs.time
      } else {
        stop("obs.time must be a vector of length 2 or a matrix with n.indiv rows and 2 columns")
      }

      sim_data <- datagen(
        i = i,
        n.indiv = n.indiv,
        a01 = haz["a01"],
        a02 = haz["a02"],
        a12 = haz["a12"],
        shape01 = haz["shape01"],
        shape02 = haz["shape02"],
        shape12 = haz["shape12"],
        beta01.sex = haz["beta01.sex"],
        beta02.sex = haz["beta02.sex"],
        beta12.sex = haz["beta12.sex"],
        beta01.age = haz["beta01.age"],
        beta02.age = haz["beta02.age"],
        beta12.age = haz["beta12.age"],
        sex = sex,
        age_raw = age_raw,
        age.center.scale = age.center.scale,
        obs_times = obs_times,
        max.time = max.time
      )

      # Process Full data
      full_dat <- sim_data$fulldata
      full_dat$epoch <- ep
      full_dat$replicate <- i
      full_dat$scheme <- "Full"
      full_dat$visit1 <- obs_times[, 1]
      full_dat$visit2 <- obs_times[, 2]
      full_dat$a01 <- haz["a01"]
      full_dat$a02 <- haz["a02"]
      full_dat$a12 <- haz["a12"]
      full_dat$shape01 <- haz["shape01"]
      full_dat$shape02 <- haz["shape02"]
      full_dat$shape12 <- haz["shape12"]
      full_dat$beta01.sex <- haz["beta01.sex"]
      full_dat$beta02.sex <- haz["beta02.sex"]
      full_dat$beta12.sex <- haz["beta12.sex"]
      full_dat$beta01.age <- haz["beta01.age"]
      full_dat$beta02.age <- haz["beta02.age"]
      full_dat$beta12.age <- haz["beta12.age"]

      # Process CensVisit data
      censvisit_dat <- sim_data$CensVisit
      censvisit_dat$epoch <- ep
      censvisit_dat$replicate <- i
      censvisit_dat$scheme <- "CensVisit"
      censvisit_dat$a01 <- haz["a01"]
      censvisit_dat$a02 <- haz["a02"]
      censvisit_dat$a12 <- haz["a12"]
      censvisit_dat$shape01 <- haz["shape01"]
      censvisit_dat$shape02 <- haz["shape02"]
      censvisit_dat$shape12 <- haz["shape12"]
      censvisit_dat$beta01.sex <- haz["beta01.sex"]
      censvisit_dat$beta02.sex <- haz["beta02.sex"]
      censvisit_dat$beta12.sex <- haz["beta12.sex"]
      censvisit_dat$beta01.age <- haz["beta01.age"]
      censvisit_dat$beta02.age <- haz["beta02.age"]
      censvisit_dat$beta12.age <- haz["beta12.age"]

      all_results[[length(all_results) + 1]] <- full_dat
      all_results[[length(all_results) + 1]] <- censvisit_dat
    }
  }
  
  out <- do.call(rbind, all_results)
  rownames(out) <- NULL
  return(out)
}

####################
# Example usage
####################
#
# # Define priors for baseline hazards (using gamma distributions)
#
# mean_sd_to_gamma <- function(mean, sd) {
#   var <- sd^2
#   alpha <- mean^2 / var
#   beta <- mean / var
#   list(alpha = alpha, beta = beta)
# }
#
# # For beta parameters (covariate effects), you provided point estimates
# # We'll use these as means with reasonable standard deviations (e.g., 0.1)
# # This allows variation in the simulations
#
# priors <- list(
#   epoch1 = list(
#     # Baseline hazard parameters (Gamma distributions)
#     a01 = mean_sd_to_gamma(mean = 2.154361e-05, sd = 1.107482e-05),
#     a02 = mean_sd_to_gamma(mean = 8.458898e-05, sd = 1.038512e-05),
#     a12 = mean_sd_to_gamma(mean = 0.0008532299, sd = 0.0005502242),
#
#     # Shape parameters (assume shape ~ 1 for exponential-like behavior)
#     # If you have better estimates, replace these
#     shape01 = list(alpha = 4, beta = 4),    # mean = 1, sd = 0.5
#     shape02 = list(alpha = 4, beta = 4),    # mean = 1, sd = 0.5
#     shape12 = list(alpha = 4, beta = 4),    # mean = 1, sd = 0.5
#
#     # Covariate effects - age (Normal distributions)
#     beta01_age = list(mean = 0.24599230, sd = 0.1),
#     beta02_age = list(mean = -0.61220429, sd = 0.1),
#     beta12_age = list(mean = -0.02149175, sd = 0.1),
#
#     # Covariate effects - sex (Normal distributions)
#     beta01_sex = list(mean = 0.13965049, sd = 0.1),
#     beta02_sex = list(mean = 0.08236421, sd = 0.1),
#     beta12_sex = list(mean = 0.04211837, sd = 0.1)
#   )
# )
#
# # Run simulation for Epoch 1 (Framingham 1948-1974)
# result <- simulate_from_priors_df(
#   priors = priors["epoch1"],
#   N = 100,                          # Number of replicates
#   n.indiv = 2300,                   # Sample size
#   sex.prob = 0.58,                  # P(female)
#   mean.age = 69,                    # Mean age
#   sd.age = 7.235,                   # SD of age
#   age.center.scale = TRUE,          # Standardize age
#   obs.time = c(2, 4.5) * 400,       # Visit times (scaled)
#   beta01.sex = 0.2706335,           # Sex effect on 0→1
#   beta02.sex = -0.07849348,         # Sex effect on 0→2
#   beta12.sex = -0.06152487,         # Sex effect on 1→2
#   beta01.age = 0.1395154,           # Age effect on 0→1
#   beta02.age = 0.02122299,          # Age effect on 0→2
#   beta12.age = 0.06483482,          # Age effect on 1→2
#   ncores = 1                        # Number of cores for parallel processing
# )
#
# # View results
# head(result)
# table(result$epoch, result$scheme)

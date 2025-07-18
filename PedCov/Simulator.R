library(dplyr)
library(stringr)
library(actuar)

data_simulation <- function(variant=NULL, par=NULL) {
  
  d <- read.table(paste0("PedCov/pedcovid_data_structure_", variant, ".txt"), header=T) %>%
    group_by(id_hh) %>%
    mutate(hh_size=n()) %>%
    ungroup() %>%
    select(-c("conf","npi_stop","duration_npi"))
  
  d <- d[rep(1:nrow(d),5),] # duplicate database
  d <- group_by(d, id_hh, id_patient) %>%
    mutate(id_rep = row_number(), .after="id_hh") %>%
    ungroup() %>%
    mutate(id_hh = as.numeric(str_extract(id_hh, "[0-9]*"))) %>%
    mutate(id_patient=row_number()) %>%
    group_by(id_hh, id_rep) %>%
    arrange(id_hh, .by_group = T) %>%
    mutate(id_hh = cur_group_id()) %>%
    ungroup()
  
  delayDist <- readRDS(paste0("PedCov/Delays_final_",variant,".rds"))
  
  pb <- txtProgressBar(min=1, max=length(unique(d$id_hh)), style=3) # Progression bar to be drawn on console
  # message(paste("Simulating", variant, "households..."))
  
  n_asympto_non_detect <- 0
  no_positive_test <- 0
  new_data_hh_list <- vector("list", length(unique(d$id_hh)))
  for(hh in unique(d$id_hh)) {
    
    # setTxtProgressBar(pb, hh)
    # message(paste("Creating household",hh,"over",max(d$id_hh)))
    
    d_hh <- d[d$id_hh==hh,] # Keep only hh data
    new_data_hh <- simulate_outbreak(d_hh, variant, delayDist, par) # Simulate outbreak
    
    # Outbreak simulation can produce households where none has a positive test
    # Those households are not usable, so we need to re-simulate. We keep track of the times it happens
    while(sum(is.na(new_data_hh$infect_status))==dim(d_hh)[1]) {
      new_data_hh <- simulate_outbreak(d_hh, variant, delayDist, par)
      no_positive_test <- no_positive_test+1
    }
    # Some asymptomatic individuals do not have a "date_sympt" because they may not have a positive test during the study (if inclusion is too far from infection)
    # Then, we consider them as not infected, and we keep count
    n_asympto_non_detect <- n_asympto_non_detect + length(new_data_hh$id_patient[new_data_hh$infect_status>0 & new_data_hh$date_sympt==1000])
    new_data_hh <- mutate(new_data_hh, infect_status = ifelse(date_sympt==1000, 0, infect_status),
                          first_test_pos = ifelse(date_sympt==1000, 1000, first_test_pos),
                          last_test_neg = ifelse(date_sympt==1000, -1000, last_test_neg))
    
    new_data_hh_list[[which(unique(d$id_hh)==hh)]] <- new_data_hh
  }
  close(pb) # Close progress bar
  
  new_data <- data.table::rbindlist(new_data_hh_list, fill=TRUE) # add the hh to the dataframe
  # message("-------DIAGNOSTICS-------")
  # message(paste("Re simulate because no positive test:", no_positive_test))
  # message(paste("Undetected asympto in the dataset:", n_asympto_non_detect))
  return(select(new_data, !id_rep))
}

simulate_outbreak <- function(d_hh=NULL, variant=NULL, delayDist=NULL, par=NULL) {
  
  if(variant=="alpha") {
    p_asympto <- 0.4
    p_test <- 1/21
    
    mIncub <- 4.42
    sdIncub <- 2.3
    #minIncub <- 2
    #maxIncub <- 7

    shapeInf <- 2
    scaleInf <- 1/0.44
  } else if(variant=="omicron"){
    p_asympto <- 0.3
    p_test <- 1/14
    
    mIncub <- 3.09
    sdIncub <- 1.64
    #minIncub <- 1
    #maxIncub <- 5
    
    shapeInf <- 3.531
    scaleInf <- 1/1.098
  }
  shapeIncub = mIncub^2/sdIncub^2
  scaleIncub = sdIncub^2/mIncub
  # Scale discretized distribution
  genTime_distrib_discretized <- discretize(pgamma(x, shape=shapeInf, scale=scaleInf), from=0, to=1000, step=1, method="rounding") / (1-pgamma(1, shape=shapeInf, scale=scaleInf))
  
  d_hh <- mutate(d_hh,
                 inf_date=NA,
                 incl_dt=NA)
  # Set the minimum duration of followup (it will be added to the first positive test/symptoms date when they come up)
  n_ite <- unique(d_hh$duration_followup)
  
  ############## Create lists for test dates and test results (each element is a dataframe of test dates and results for one individual)
  tests <- setNames(vector("list", length(d_hh$id_patient)), d_hh$id_patient)
  
  ################## INDEX CASE ##################
  # Randomly chose index case
  index <- sample(d_hh$id_patient, 1)
  d_hh$is_index[d_hh$id_patient==index] <- 1

  # Attribute infection date
  d_hh$inf_date[d_hh$id_patient==index] <- 0
  
  # Determine infection status
  if(rbinom(1, 1, p_asympto)==1) { # index is asymptomatic
    d_hh$infect_status[d_hh$id_patient==index] <- 2
  } else { # index is symptomatic
    d_hh$infect_status[d_hh$id_patient==index] <- 1
    # Determine symptoms onset date
    incub <- round(rgamma(1, shape=shapeIncub, scale=scaleIncub))
    d_hh$date_sympt[d_hh$id_patient==index] <- d_hh$inf_date[d_hh$id_patient==index] + incub
  }
  
  # Add npi_stop to all individuals at first iteration where someone has symptoms/1st test
  if( d_hh$date_sympt[d_hh$id_patient==index]%in%0 ) {
    n_ite <- min(d_hh$date_sympt, na.rm=T) + n_ite
  }
  
  ################## PROPAGATE ##################
  # id of susceptible individuals
  susc <- d_hh$id_patient[d_hh$id_patient!=index]

  ### For each timestep
  t <- 0
  # n_ite might change along the way (when inclusion happens if index is not already inclusion)
  while(t<n_ite) {
    t <- t+1
    
    ## For each susceptible individual in household
    for(ind in susc) {
      # Compute probability of infection of ind at time t
      proba_inf <- 1-exp(-compute_foi(d_hh, ind, susc, t, variant, genTime_distrib_discretized, par))
      
      # If ind is infected
      if(rbinom(1, 1, proba_inf)==1) {

        # Attribute infection date on this iteration
        d_hh$inf_date[d_hh$id_patient==ind] <- t
        
        # Attribute infection status
        if(rbinom(1, 1, p_asympto)==1) { # if is asymptomatic
          d_hh$infect_status[d_hh$id_patient==ind] <- 2
        } else { # if is symptomatic
          d_hh$infect_status[d_hh$id_patient==ind] <- 1
          # Determine symptoms onset date
          incub <- round(rgamma(1, shape=shapeIncub, scale=scaleIncub))
          d_hh$date_sympt[d_hh$id_patient==ind] <- t + incub
        }
        
      } # if ind does not get infected, do not do anything (by default, ind is not infected)
    }
    
    # If symptoms onset at t, then plan a test for all household with a certain delay
    if(t %in% d_hh$date_sympt) {
      
      sympto_id <- d_hh$id_patient[d_hh$date_sympt%in%t]

      # Determine date of test for symptomatic individuals
      new_test <- t + rgeom(1, prob=delayDist$delaySymptToTest)

      # Set result of test for all individuals with symptoms onset at time t
      testpos <- F # marker for "at least one positive test made"
      for(id in sympto_id) {
        id_char <- as.character(id) # tests list takes character arguments
        
        if( d_hh$inf_date[d_hh$id_patient==id]<(new_test-1) & new_test<d_hh$inf_date[d_hh$id_patient==id]+15 ) {
          tests[[id_char]] <- rbind(tests[[id_char]], data.frame(date=new_test, result=1))
          testpos <- T
        } else tests[[id_char]] <- rbind(tests[[id_char]], data.frame(date=new_test, result=0))
        
      }
      
      # If at least one test is positive, plan testing for other household members a few days later (no test result for now)
      if(testpos) {
        tests <- plan_family_testing(tests, d_hh, new_test, sympto_id, delayDist)
      }
      
      # If n_ite is not updated yet, set it (first symptoms onset + duration followup)
      if(n_ite==unique(d_hh$duration_followup)) {
        n_ite <- t + n_ite
      }
      
    }
    
    # If there is at least a test at time t that does not have a result (a planned family member test), compute result
    for( id_char in names(tests)[sapply(tests, function(df) any(df$date==t & is.na(df$result)) )] ) {
      
      id <- as.numeric(id_char)
      # If individual has been infected and test date falls into positivity period
      if( !is.na(d_hh$inf_date[d_hh$id_patient==id]) && d_hh$inf_date[d_hh$id_patient==id]<(t-1) & t<d_hh$inf_date[d_hh$id_patient==id]+15 ) {
        tests[[id_char]]$result[tests[[id_char]]$date%in%t] <- 1

        # If n_ite is not updated yet, set it (first symptoms onset + duration followup)
        if(n_ite==unique(d_hh$duration_followup)) {
          n_ite <- t + n_ite
        }
        
      } else tests[[id_char]]$result[tests[[id_char]]$date%in%t] <- 0
      
    }
    
    # If there is no test at time t, each individual has a proba p_test of testing
    if( !any(sapply(tests, function(x) t %in% x$date)) ) {

      testpos_id <- NULL # marker for individuals who will have made a random test at t
      for(id in d_hh$id_patient) {
        
        if(rbinom(1, 1, p_test)) {
          
          id_char <- as.character(id)
          # If individual has been infected and test date falls into positivity period
          if( !is.na(d_hh$inf_date[d_hh$id_patient==id]) && d_hh$inf_date[d_hh$id_patient==id]<(t-1) & t<d_hh$inf_date[d_hh$id_patient==id]+15 ) {
            tests[[id_char]] <- rbind(tests[[id_char]], data.frame(date=t, result=1))
            testpos_id <- c(testpos_id, id)
            
            # If n_ite is not updated yet, set it (first symptoms onset + duration followup)
            if(n_ite==unique(d_hh$duration_followup)) {
              n_ite <- t + n_ite
            }
            
          } else tests[[id_char]] <- rbind(tests[[id_char]], data.frame(date=t, result=0))
        }
      }
      # If at least one test is positive, plan testing for other household members a few days later (no test result for now)
      if(length(testpos_id)>0) {
        tests <- plan_family_testing(tests, d_hh, t, testpos_id, delayDist)
      }
    }
    
    susc <- d_hh$id_patient[d_hh$infect_status==0]
    
    # If there are no more susceptible individuals in the hh, get out of the time loop
    if(length(susc)==0) break
  }
  
  ################## ADD INCLUSION ##################
  # Choose inclusion case among individuals with a positive test
  # Find id of individuals with a positive test
  positive_ids <- names(tests)[sapply(tests, function(df) 1%in%df$result)]
  # If no individual has a positive test, exit the simulation
  if(length(positive_ids)==0) {
    d_hh$infect_status <- NA # marks a non valid simulation
    return(d_hh) # exits the simulation
  } else inclu <- as.numeric(sample(positive_ids, 1))
  d_hh$is_incluCase[d_hh$id_patient==inclu] <- 1

  # Inclusion date is after first positive test by inclusion case
  t1 <- min(tests[[as.character(inclu)]]$date[tests[[as.character(inclu)]]$result%in%1])
  d_hh$incl_dt <- t1 + rpois(1, lambda=delayDist$delayTestInclu)

  # Add followup tests
  followup_dates <- unique(d_hh$incl_dt) + c(0,3,7,15,45)
  tests <- lapply(tests, function(df) rbind(df, data.frame(date=followup_dates, result=rep(NA,5))))
  
  # Add first_test_pos and last_test_neg to all individuals + date_sympt for asymptomatic
  for( id in d_hh$id_patient[d_hh$infect_status>0] ) {
    d_hh <- add_first_last_tests(d_hh, id, tests)
  }
  
  # Reformat symptoms date so the first one is 30
  m <- min(c(d_hh$date_sympt, d_hh$incl_dt, d_hh$inf_date, d_hh$npi_stop, d_hh$first_test_pos, d_hh$last_test_neg), na.rm=T)
  # Attribute end of followup
  d_hh <- mutate(d_hh, date_sympt = case_when( date_sympt==1000 ~ 1000,
                                               date_sympt!=1000 ~ 30 + round(as.numeric(date_sympt - m)) ),
                 incl_dt = case_when( is.na(incl_dt) ~ NA,
                                      !is.na(incl_dt) ~ 30 + round(as.numeric(incl_dt - m)) ),
                 inf_date = case_when( is.na(inf_date) ~ NA,
                                       !is.na(inf_date) ~ 30 + round(as.numeric(inf_date - m)) ),
                 first_test_pos = case_when( is.na(first_test_pos) ~ 1000,
                                             !is.na(first_test_pos) ~ 30 + round(as.numeric(first_test_pos - m)) ),
                 last_test_neg = case_when( is.na(last_test_neg) ~ -1000,
                                            !is.na(last_test_neg) ~ 30 + round(as.numeric(last_test_neg - m)) ),
                 end_followup = round(min(date_sympt, na.rm=T)+duration_followup) )

  return(d_hh)#select(d_hh, id_patient, id_hh, hh_size, date_sympt, infect_status, end_followup, age, protected, age_exact))
}

# Plan a test for family members a few days after symptoms or positive test
# t = date from which to plan a test (symptoms onset or positive test)
# id_testPos = id of individuals for whom not to plan a test
plan_family_testing <- function(tests=NULL, d_hh=NULL, t=NULL, id_testPos=NULL, delayDist=NULL) {
  
  other_hh_members <- as.character(d_hh$id_patient[!(d_hh$id_patient%in%id_testPos)])
  new_test_hh <- t + rgeom(1, prob=delayDist$delay_to_minTestPos)
  tests_skipped <- rbinom(length(other_hh_members), 1, delayDist$propMissing_byHHsize$prop_missing[delayDist$propMissing_byHHsize$hsize==length(d_hh$id_patient)])
  tests[other_hh_members[which(tests_skipped==0)]] <- lapply(tests[other_hh_members[which(tests_skipped==0)]], function(df) rbind(df, data.frame(date=new_test_hh, result=NA)))

  return(tests)
}

# Define first and last test dates
add_first_last_tests <- function(d_hh=NULL, ind=NULL, tests=NULL) {
  
  # Compute tests results
  tests_i <- tests[[as.character(ind)]]
  for( d in tests_i$date[is.na(tests_i$result)] ){
    if( d_hh$inf_date[d_hh$id_patient==ind]<(d-1) & d<d_hh$inf_date[d_hh$id_patient==ind]+15 ) tests_i$result[tests_i$date==d] <- 1
    else tests_i$result[tests_i$date==d] <- 0
  }
  tests_i_pos <- filter(tests_i, result==1)
  tests_i_neg <- filter(tests_i, result==0)
  
  # Add first test pos and last test neg
  # If individual is symptomatic, first_test_pos is first test pos
  # If individual is asymptomatic, first_test_pos will be LAST TEST POS
  if(d_hh$infect_status[d_hh$id_patient==ind]==1) {
    d_hh$first_test_pos[d_hh$id_patient==ind] <- min(tests_i_pos$date, na.rm=T)
  } else if(d_hh$infect_status[d_hh$id_patient==ind]==2) {
    d_hh$date_sympt[d_hh$id_patient==ind] <- min(tests_i_pos$date, na.rm=T)
    if(is.infinite(d_hh$date_sympt[d_hh$id_patient==ind])) d_hh$date_sympt[d_hh$id_patient==ind] <- 1000
    d_hh$first_test_pos[d_hh$id_patient==ind] <- max(tests_i_pos$date, na.rm=T)
  }
  d_hh$last_test_neg[d_hh$id_patient==ind] <- max(tests_i_neg$date[tests_i_neg$date<d_hh$date_sympt[d_hh$id_patient==ind]], na.rm=T)
  
  if(is.infinite(d_hh$first_test_pos[d_hh$id_patient==ind])) d_hh$first_test_pos[d_hh$id_patient==ind] <- NA
  if(is.infinite(d_hh$last_test_neg[d_hh$id_patient==ind])) d_hh$last_test_neg[d_hh$id_patient==ind] <- NA
  
  # It can happen that the there is a negative test after the positive test if it is more than 15 days after infection (with followup visit 15 for example)
  # In that case, it is not a last negative test it is a first negative after infection and we do not care about it
  if(!is.na(d_hh$first_test_pos[d_hh$id_patient==ind])
     && !is.na(d_hh$last_test_neg[d_hh$id_patient==ind])
     && d_hh$last_test_neg[d_hh$id_patient==ind]>d_hh$first_test_pos[d_hh$id_patient==ind] ) d_hh$last_test_neg[d_hh$id_patient==ind] <- NA
  
  return(d_hh)
}

compute_foi <- function(data=NULL, ind=NULL, susc=NULL, t=NULL, variant=NULL, genTime_distrib_discretized=NULL, par=NULL) {
  
  alpha <- par$alpha #0.001
  beta <- par$beta
  delta <- par$delta #1.3
  mu_inf <- par$mu_inf # c(SI = 1, SC= 1, AI = 1, AC = 1, AA = 1)
  mu_susc <- par$mu_susc # c(I = 1, C = 1)
  mu_protect <- par$mu_protect

  infecteds <- data$id_patient[!(data$id_patient %in% susc)]
  beta_foyer <- 0
  
  for(ix in infecteds) {
    
    generation_time <- t - data$inf_date[data$id_patient==ix]
    proba_inf_ix_ind <- genTime_distrib_discretized[generation_time+1] # Already scaled
    
    # It should not happen but in the case when generation time is 0, dgamma gives +inf but proba should be 0. We skip directly to the next infected individual
    if(is.infinite(proba_inf_ix_ind)) next
    
    # Susceptibility of ind depending on age and symptoms
    if(data$age[data$id_patient==ix]==1 & data$infect_status[data$id_patient==ix]==1) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['SC'] # sympt children
    else if(data$age[data$id_patient==ix]==0 & data$infect_status[data$id_patient==ix]==1) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['SI'] # sympt adults
    else if(data$age[data$id_patient==ix]==0 & data$infect_status[data$id_patient==ix]==2) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['AI'] # asympt infants
    else if(data$age[data$id_patient==ix]==1 & data$infect_status[data$id_patient==ix]==2) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['AC'] # asympt children
    else if(data$age[data$id_patient==ix]==2 & data$infect_status[data$id_patient==ix]==2) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['AA'] # asympt adults
    # Infectivity of ix depending on protection
    if(data$protected[data$id_patient==ix]==1) proba_inf_ix_ind <- proba_inf_ix_ind *mu_protect['transm']
    
    beta_foyer <- beta_foyer + proba_inf_ix_ind
  }
  
  # Susceptibility of ind depending on age
  if(data$age[data$id_patient==ind]==1) beta_foyer <- beta_foyer *mu_susc['C'] # children
  else if(data$age[data$id_patient==ind]==0) beta_foyer <- beta_foyer *mu_susc['I'] # adults
  # Susceptibility of ind depending on protection
  if(data$protected[data$id_patient==ind]==1) beta_foyer <- beta_foyer *mu_protect['acq']
  
  proba_inf <- alpha + beta / (data$hh_size[data$id_patient==ind]/4)^delta * beta_foyer
  
  return(proba_inf)
}


data_selection <- function(d=NULL, variant=NULL, method=NULL) {
  
  # Total nb of households
  if(variant%in%"alpha") tot_hh <- 128
  else if(variant%in%"omicron") tot_hh <- 54
  
  # available households
  hh <- unique(d$id_hh)
  recruit <- NULL
  recruit_size <- 0
  
  while(recruit_size < tot_hh) { # while there is not enough hh is the final base
    
    u <- sample(hh, 1) # pick one randomly
    
    if(method=="pedcov") {
      # If inclusion case is an adult --> exclude and go to next iteration
      if(d$age_exact[d$id_hh==u & d$is_incluCase==1] > 18 ) {
        hh <- hh[hh!=u] # Remove this hh from the list of pickable hh
        next
      }
    }
    else if(method=="adultcov") {
      # If inclusion case is not an adult --> exclude and go to next iteration
      if(d$age_exact[d$id_hh==u & d$is_incluCase==1] < 18 ) {
        hh <- hh[hh!=u] # Remove this hh from the list of pickable hh
        next
      }
    }
    # If inclusion case has not had symptoms or test yet --> exclude
    if( min(d$date_sympt[d$id_hh==u & d$is_incluCase==1], d$first_test_pos[d$id_hh==u & d$is_incluCase==1], na.rm=T) >= d$incl_dt[d$id_hh==u & d$is_incluCase==1]) {
      hh <- hh[hh!=u] # Remove this hh from the list of pickable hh
      next
    }
    recruit <- rbind(recruit, d[d$id_hh==u,]) # keep it
    recruit_size <- recruit_size + 1 # increase count
    
    hh <- hh[hh!=u] # Remove this hh from the list of pickable hh
  }

  return(recruit)
}

simulate <- function(par, variant, select_process, id=1) {

  if(select_process=="all") {
    sim_data <- data_simulation(variant, par)
    recruit_pedcov <- mutate(data_selection(sim_data, variant, "pedcov"), "variant"=variant) %>%
    mutate(id_simu = id, select_process = "pedcov")

    recruit_adultcov <- mutate(data_selection(sim_data, variant, "adultcov"), "variant"=variant) %>%
    mutate(id_simu = id, select_process = "adultcov")

    recruit_random <- mutate(data_selection(sim_data, variant, "random"), "variant"=variant) %>%
    mutate(id_simu = id, select_process = "random")

    recruit <- rbind(recruit_pedcov, recruit_adultcov, recruit_random)
  }
  else {
    sim_data <- data_simulation(variant, par)
    recruit <- mutate(data_selection(sim_data, variant, select_process), "variant"=variant) %>%
    mutate(id_simu = id, select_process = select_process)
  }

  return(select(recruit, -c("inf_date","incl_dt")))
}

# p <- list("beta" = 0.5,
#             "alpha" = 0.002,
#             "delta" = 1.3,
#             "mu_inf" = c(SI = 1, SC= 1, AI = 1, AC = 1, AA = 1),
#             "mu_susc" = c(I = 1, C = 1),
#             "mu_protect" = c("acq"=1, "transm"=1))
# t <- simulate(p, variant="alpha", select_process="random")


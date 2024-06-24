library(dplyr)
library(stringr)
library(actuar)


data_simulation <- function(variant=NULL, par=NULL, verbose=F) {
  
  d <- read.table(paste0("Simulator/pedcovid_data_structure_", variant, ".txt"), header=T)
  
  d <- d[rep(rep(1:nrow(d),50)),] # duplicate database
  d <- group_by(d, id_hh, id_patient) %>%
    mutate(id_rep = row_number(), .after="id_hh") %>%
    ungroup() %>%
    mutate(id_hh = as.numeric(str_extract(id_hh, "[0-9]*"))) %>%
    mutate(id_patient=row_number()) %>%
    group_by(id_hh, id_rep) %>%
    arrange(id_hh, .by_group = T) %>%
    mutate(test = cur_group_id(), .after="id_rep") %>%
    mutate(id_hh = paste0(cur_group_id(), "-", str_sub(variant,1,1))) %>%
    ungroup()
  
  delayDist <- readRDS(paste0("Simulator/Delays_",variant,".rds"))
  
  #pb <- txtProgressBar(min=1, max=length(unique(d$id_hh)), style=3) # Progression bar to be drawn on console
  #message(paste("Simulating", variant, "households..."))
  
  new_data <- NULL
  for(hh in unique(d$id_hh)) {
    
    #setTxtProgressBar(pb, as.numeric(str_split_1(hh,"-")[1]))
    
    d_hh <- d[d$id_hh==hh,] # Keep only hh data
    new_data_hh <- simulate_outbreak(d_hh, variant, delayDist, par, verbose) # Simulate outbreak
    
    # Add variable for category for future selection of hh
    if( sum(c(1,2) %in% new_data_hh$infect_status[new_data_hh$age_exact<19])==0 ) { # If no infected child
      first_i <- "no_inf_child"
    } else { # If at least one infected child
      min_date_ch <- min(new_data_hh$date_sympt[new_data_hh$age_exact<19 & new_data_hh$infect_status!=0]) # date of first sympt_date in infected children
      first_i_status <- new_data_hh$infect_status[new_data_hh$age_exact<19 & new_data_hh$date_sympt==min_date_ch] # status of this child
      if( length(first_i_status)>1 ) first_i <- "both" # may have 2 children with same date
      else if(first_i_status==1) first_i <- "sympto_child"
      else if(first_i_status==2) first_i <- "asympto_child"
    }
    new_data_hh <- mutate(new_data_hh, first_inf=first_i) # add this column to the hh
    
    new_data <- rbind(new_data, new_data_hh) # add the hh to the dataframe
    
  }
  #close(pb) # Close progress bar
  
  return(new_data)
}

simulate_outbreak <- function(d_hh=NULL, variant=NULL, delayDist=NULL, par=NULL, verbose=F) {
  
  if(variant=="alpha") {
    p_asympto <- 0.22
    
    mIncub <- 4.42
    sdIncub <- 2.3
    minIncub <- 2
    maxIncub <- 7
    
    shapeInf <- 2
    scaleInf <- 1/0.44
  } else if(variant=="omicron"){
    p_asympto <- 0.25
    
    mIncub <- 3.09
    sdIncub <- 1.64
    minIncub <- 1
    maxIncub <- 5
    
    shapeInf <- 3.531
    scaleInf <- 1/1.098
  }
  shapeIncub = mIncub^2/sdIncub^2
  scaleIncub = sdIncub^2/mIncub
  genTime_distrib_discretized <- discretize(pgamma(x, shape=shapeInf, scale=scaleInf), from=0, to=1000, step=1, method="rounding")
  
  d_hh <- mutate(d_hh, inf_date=NA,
                 incl_dt=NA)
  
  if(verbose) message("Choice of index")
  ################## INDEX CASE ##################
  # Randomly chose index case
  index <- sample(d_hh$id_patient, 1)
  d_hh$is_index[d_hh$id_patient==index] <- 1
  if(verbose) message(paste("   Index is", index))
  
  # Attribute infection date
  d_hh$inf_date[d_hh$id_patient==index] <- 0
  
  # Determine infection status and symptoms date
  u <- runif(1,0,1)
  if(u < p_asympto) { # index is asymptomatic
    d_hh$infect_status[d_hh$id_patient==index] <- 2
    incub <- round(runif(1, minIncub, maxIncub))
  } else { # index is symptomatic
    d_hh$infect_status[d_hh$id_patient==index] <- 1
    incub <- round(rgamma(1, shape=shapeIncub, scale=scaleIncub))
  }
  d_hh$date_sympt[d_hh$id_patient==index] <- d_hh$inf_date[d_hh$id_patient==index] + incub
  if(verbose) message(paste("   Index infection status is", d_hh$infect_status[d_hh$id_patient==index]))
  if(verbose) message(paste("   Index infection date is", d_hh$inf_date[d_hh$id_patient==index]))
  
  
  ################## PROPAGATE ##################
  # id of susceptible individuals
  susc <- d_hh$id_patient[d_hh$id_patient!=index]
  if(verbose) message(paste("Id of susceptible individuals are: ", paste(susc, collapse=" and ")))
  
  n_ite <- unique(d_hh$duration_followup)
  
  ### For each timestep
  for(t in c(1:n_ite)) {
    
    ## For each susceptible individual in household
    for(ind in susc) {
      # Compute probability of infection of ind at time t
      proba_inf <- compute_foi(d_hh, ind, susc, t, variant, genTime_distrib_discretized, par, verbose)
      
      u <- runif(1,0,1)
      if(u < proba_inf) { # if ind gets infected
        
        # Attribute infection date on this iteration
        d_hh$inf_date[d_hh$id_patient==ind] <- t
        
        # Attribute infection status, symptoms date and first positive test
        u <- runif(1,0,1)
        if(u < p_asympto) { # if is asympto
          d_hh$infect_status[d_hh$id_patient==ind] <- 2 # infect_status = 2
          incub <- round(runif(1, minIncub, maxIncub)) # sympt_date = inf_date + incub
        } else { # if is sympto
          d_hh$infect_status[d_hh$id_patient==ind] <- 1 # infect_status = 1
          incub <- round(rgamma(1, shape=shapeIncub, scale=scaleIncub)) # sympt_date = inf_date + incub
        }
        d_hh$date_sympt[d_hh$id_patient==ind] <- t + incub
        
      } # if ind does not get infected, do not do anything (by default, ind is not infected)
    }
    
    susc <- d_hh$id_patient[d_hh$infect_status==0]
    
    # If there are no more susceptible individuals in the hh, get out of the time loop
    if(length(susc)==0) break
  }
  
  ################## ADD INCLUSION ##################
  inclu <- ifelse( length(d_hh$id_patient[d_hh$infect_status!=0])==1, # if there is only one infected in the hh (ifelse needed because sample(x= vector of one value) gives sample(1:x))
                   d_hh$id_patient[d_hh$infect_status!=0], # incl case will be the only infected
                   sample(d_hh$id_patient[d_hh$infect_status!=0], 1) ) # else, we pick randomly
  d_hh$is_incluCase[d_hh$id_patient==inclu] <- 1
  if(d_hh$infect_status[d_hh$id_patient==inclu]==1) {
    d_hh$incl_dt <- d_hh$date_sympt[d_hh$id_patient==inclu] + rgamma(1, shape=delayDist[1], rate=delayDist[2]) - ifelse(delayDist[3]<=0, abs(delayDist[3])+1, 0)
  } else if(d_hh$infect_status[d_hh$id_patient==inclu]==2) {
    d_hh$incl_dt <- d_hh$date_sympt[d_hh$id_patient==inclu] + rgamma(1, shape=delayDist[4], rate=delayDist[5]) - ifelse(delayDist[6]<=0, abs(delayDist[6])+1, 0)
  }
  
  
  
  # Reformat symptoms date so the first one is 30
  m <- min(c(d_hh$date_sympt, d_hh$incl_dt, d_hh$inf_date), na.rm=T)
  # Attribute end of followup
  d_hh <- mutate(d_hh, date_sympt = case_when( date_sympt==1000 ~ 1000,
                                               date_sympt!=1000 ~ 30 + as.numeric(date_sympt - m) ),
                 # date_sympt!=1000 ~ 30 + as.numeric(date_sympt - min(c(date_sympt,incl_dt,inf_date), na.rm=T)) ),
                 incl_dt = case_when( is.na(incl_dt) ~ NA,
                                      !is.na(incl_dt) ~ 30 + as.numeric(incl_dt - m) ),
                 # !is.na(incl_dt) ~ 30 + as.numeric(incl_dt - min(c(date_sympt,incl_dt,inf_date), na.rm=T)) ),
                 inf_date = case_when( is.na(inf_date) ~ NA,
                                       !is.na(inf_date) ~ 30 + as.numeric(inf_date - m) ),
                 # !is.na(inf_date) ~ 30 + as.numeric(inf_date - min(c(date_sympt,incl_dt,inf_date), na.rm=T)) ),
                 end_followup = min(date_sympt)+duration_followup )
  
  return(d_hh)#select(d_hh, id_patient, id_hh, hh_size, date_sympt, infect_status, end_followup, age, protected, age_exact))
  
}

compute_foi <- function(data=NULL, ind=NULL, susc=NULL, t=NULL, variant=NULL, genTime_distrib_discretized=NULL, par=NULL, verbose=F) {
  alpha <- par$alpha
  beta <- par$beta
  delta <- par$delta
    
  mu_inf <- par$mu_inf #c(SC = 1, SA= 1, AI = 1, AC = 1, AA = 1)
  mu_susc <- par$mu_susc #c(C = 1, A = 1)
  mu_protect <- par$mu_protect #c(acq = 0.8, transm = 0.8)
  
  infecteds <- data$id_patient[!(data$id_patient %in% susc)]
  
  beta_foyer <- 0
  
  for(ix in infecteds) {
    
    generation_time <- t - data$inf_date[data$id_patient==ix]
    proba_inf_ix_ind <- genTime_distrib_discretized[generation_time+1] / (1-genTime_distrib_discretized[1])
    
    # It should not happen but in the case when generation time is 0, dgamma gives +inf but proba should be 0. We skip directly to the next infected individual
    if(is.infinite(proba_inf_ix_ind)) next
    
    # Susceptibility of ind depending on age and symptoms
    if(data$age[data$id_patient==ix]==1 & data$infect_status[data$id_patient==ix]==1) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['SC'] # sympt children
    else if(data$age[data$id_patient==ix]==2 & data$infect_status[data$id_patient==ix]==1) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['SA'] # sympt adults
    else if(data$age[data$id_patient==ix]==0 & data$infect_status[data$id_patient==ix]==2) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['AI'] # asympt infants
    else if(data$age[data$id_patient==ix]==1 & data$infect_status[data$id_patient==ix]==2) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['AC'] # asympt children
    else if(data$age[data$id_patient==ix]==2 & data$infect_status[data$id_patient==ix]==2) proba_inf_ix_ind <- proba_inf_ix_ind *mu_inf['AA'] # asympt adults
    # Infectivity of ix depending on protection
    if(data$protected[data$id_patient==ix]==1) proba_inf_ix_ind <- proba_inf_ix_ind *mu_protect['transm']
    
    if(verbose) message(paste0("         Contribution of individual ", ix,": ", proba_inf_ix_ind))
    beta_foyer <- beta_foyer + proba_inf_ix_ind
  }
  
  # Susceptibility of ind depending on age
  if(data$age[data$id_patient==ind]==1) beta_foyer <- beta_foyer *mu_susc['C'] # children
  else if(data$age[data$id_patient==ind]==2) beta_foyer <- beta_foyer *mu_susc['A'] # adults
  # Susceptibility of ind depending on protection
  if(data$protected[data$id_patient==ind]==1) beta_foyer <- beta_foyer *mu_protect['acq']
  
  proba_inf <- alpha + beta / (data$hh_size[data$id_patient==ind]/4)^delta * beta_foyer
  
  return(proba_inf)
}


data_selection <- function(d, variant, method, verbose=F) {
  
  # Determine counts of households (total, depending on index case and depending on status of inclusion case)
  if(variant=="alpha") tot_hh <- 128
  else if(variant=="omicron") tot_hh <- 54
  tot_hh_a <- ceiling(tot_hh/5) # 1/5 of hh are included through asympto children
  tot_hh_s <- tot_hh - tot_hh_a # 4/5 of hh are included through asympto children
  if(variant=="alpha") tot_hh_inclIndex <- 88
  else if(variant=="omicron") tot_hh_inclIndex <- 44
  
  # available households
  hh <- unique(d$id_hh)
  
  if(method=="random") {
    
    sel <- sample(unique(d$id_hh), tot_hh) # just sample a total number of households
    recruit <- d[d$id_hh %in% sel,]
    
  } else {
    recruit <- NULL
    
    hh_s <- 0 # marker for count of sympto category
    hh_a <- 0 # marker for count of asympto category
    hh_inclIndex <- 0 # marker for count of inclusion=index category
    hh_inclNotIndex <- 0 # marker for count of inclusion!=index category
    not_selected <- 0
    
    while(hh_s+hh_a < tot_hh) { # while there is not enough hh is the final base
      
      u <- sample(hh, 1) # pick one randomly
      
      if(method=="pedcov") {
        
        # If inclusion case is an adult --> exclude and go to next iteration
        if(d$age_exact[d$id_hh==u & d$is_incluCase==1] > 18 ) {
          not_selected <- not_selected +1
          hh <- hh[hh!=u] # Remove this hh from the list of pickable hh
          next
        }
        
      }
      
      # If inclusion case has not had symptoms or test yet --> exclude
      if( d$date_sympt[d$id_hh==u & d$is_incluCase==1] >= d$incl_dt[d$id_hh==u & d$is_incluCase==1]) {
        not_selected <- not_selected +1
        hh <- hh[hh!=u] # Remove this hh from the list of pickable hh
        next
      }
      
      # Inclusion case is also index case (and the count for this type is not full)
      if( d$is_index[d$id_hh==u & d$is_incluCase==1] == 1 & hh_inclIndex < tot_hh_inclIndex ) {
        
        if(d$infect_status[d$id_hh==u & d$is_incluCase==1]==1 & hh_s<tot_hh_s) { # If inclusion case is symptomatic (and the count for this type is not full)
          recruit <- rbind(recruit, d[d$id_hh==u,]) # keep it
          hh_s <- hh_s+1 # increase count of sympto incl cases
          hh_inclIndex <- hh_inclIndex + 1 # increase count of incl case also index
        } else if(d$infect_status[d$id_hh==u & d$is_incluCase==1]==2 & hh_a<tot_hh_a) { # If inclusion case is asymptomatic (and the count for this type is not full)
          recruit <- rbind(recruit, d[d$id_hh==u,]) # keep it
          hh_a <- hh_a+1 # increase count of asympto incl cases
          hh_inclIndex <- hh_inclIndex + 1 # increase count of incl case also index
        } else not_selected <- not_selected +1 # If the count for this type of sympto status is already full --> exclude
        
        if(verbose) message(paste("sympto", hh_s, "asympto", hh_a))
        
        # Inclusion case is not the index case (and the count for this type is not full)
        # (ie there are household members infected visibly before inclusion case)
      } else if( d$is_index[d$id_hh==u & d$is_incluCase==1] == 0 & hh_inclNotIndex < tot_hh-tot_hh_inclIndex) {
        
        if(d$infect_status[d$id_hh==u & d$is_incluCase==1]==1 & hh_s<tot_hh_s) { # If inclusion case is symptomatic (and the count for this type is not full)
          recruit <- rbind(recruit, d[d$id_hh==u,]) # keep it
          hh_s <- hh_s+1 # increase count of sympto incl cases
          hh_inclNotIndex <- hh_inclNotIndex + 1 # increase count of incl case not index
        } else if(d$infect_status[d$id_hh==u & d$is_incluCase==1]==2 & hh_a<tot_hh_a) { # If inclusion case is asymptomatic (and the count for this type is not full)
          recruit <- rbind(recruit, d[d$id_hh==u,]) # keep it
          hh_a <- hh_a+1 # increase count of asympto incl cases
          hh_inclNotIndex <- hh_inclNotIndex + 1 # increase count of incl case not index
        } else not_selected <- not_selected +1 # If the count for this type of sympto status is already full --> exclude
        
        if(verbose) message(paste("sympto", hh_s, "asympto", hh_a))
        
      } else not_selected <- not_selected +1 # If the count for this type of incl case / index is already full --> exclude
      
      hh <- hh[hh!=u] # Remove this hh from the list of pickable hh
      
    }
  }
  
  # pedcov_recruit <- select(pedcov_recruit, id_patient, id_hh, hh_size, date_sympt, infect_status, end_followup, age, protected, age_exact)
  # write.table(pedcov_recruit, paste0("../Data/pedcovid_data_formatted_",variant,"_pedcovid.txt"), quote=F, row.names=F, col.names=F)
  
  return(recruit)
}

simulate_and_reformat <- function(par=NULL, id=1) {
  # variants: alpha, omicron
  variant <- par$variant #"alpha"
  # selection procedure: pedcov, random
  selection_procedure <- par$selection_procedure # "pedcov"
  
  # convert list
  par <- list(
    "alpha" = par$alpha,
    "beta" = par$beta,
    "delta" = par$delta,
    "mu_inf" = c(SC = par$mu_inf_SC, SA = par$mu_inf_SA, 
                 AI = par$mu_inf_AI, AC = par$mu_inf_AC, 
                 AA = par$mu_inf_AA),
    "mu_susc" = c(C = par$mu_susc_C, A = par$mu_susc_A),
    "mu_protect" = c(acq = par$mu_protect_acq, transm = par$mu_protect_transm)
  )
  
  sim_data_alpha <- data_simulation(variant, par, verbose=F)
  recruit_alpha <- data_selection(sim_data_alpha, variant, selection_procedure)
  
  recruit_pedcov <- rbind(mutate(recruit_alpha, "variant"=variant)) %>%
    mutate(id_hh = paste0(id_hh,"-p",id),
           id_simu = id,
           select_process = "pedcov-like")
  
  sim_data <- recruit_pedcov[recruit_pedcov$variant==variant,]
  
  # select columns
  sim_data <- select(sim_data, id_patient, id_hh, hh_size, date_sympt, 
                     infect_status, end_followup, age, protected, variant, 
                     id_simu, select_process)
  return(sim_data)
}

#test <- list("beta" = 0.2,
#             "mu_inf" = c(SC = 1, SA= 1, AI = 1, AC = 1, AA = 1),
#             "mu_susc" = c(C = 1, A = 1))
#t <- simulate_and_reformat(test)

# Paper Story

## Prevalence Example
- simple to show bias on simulated data set
- missingness + selection creates a joint bias we can correct for (see simulated data)
- application on real data 
- verification with C2ST, how reliable we are per epoch (only appendix)

## Visit Censoring Example
- more complex example with longitudinal data
- train model on censored and non-censored data to show bias via SBC (not only a point estimate for the bias, but checking the whole distribution)
- application on real data and comparison with other methods (naive, weibull, splines), maybe not show splines for easier comparison
- verification on real data with C2ST, show that we can detect bias if model is trained on un-censored data but applied to censored data

## PedCovid Example
- real world example with complex selection mechanism
- to illustrate the bias, show SBC with MCMC
- train one model on multiple selection mechanisms to show unbiased for all scenarios with SBC
- application on real data
- verification with C2ST again?

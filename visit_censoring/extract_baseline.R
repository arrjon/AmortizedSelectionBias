library(jsonlite)

load("/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/visit_censoring/Rdata/cumHaz_final.Rdata")
load("/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/visit_censoring/Rdata/fram_resfit.Rdata")
load("/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/visit_censoring/Rdata/baselinehazards_cox_splines.Rdata")

# Helper to extract coefficients for one epoch
extract_coefs <- function(obj, epoch_name) {
  epoch <- obj$coefs[[epoch_name]]

  swap_pairs <- c(2, 1, 4, 3, 6, 5)  # Swap pairs for Weibull and Splines to match Cox order

  data.frame(
    naive_cox   = epoch$naive_cox,
    idm_weib    = epoch$idm_weib[swap_pairs],
    idm_splines = epoch$idm_splines[swap_pairs],
    row        = rownames(epoch),
    row.names = NULL
  )
}

# Build one full epoch entry: coefs + per-transition hazards for all models
build_epoch <- function(epoch_name) {
  base    <- plot_values[[epoch_name]]
  idm     <- fram_resfit_2026$baseline[[epoch_name]]$idm
  cox_idm <- fram_resfit_2026$baseline[[epoch_name]]$cox
  cox_e   <- cox_baselines[[epoch_name]]
  spl_e   <- splines_baselines[[epoch_name]]

  build_transition <- function(tr_name) {
    tr      <- base[[tr_name]]
    cox_tr  <- cox_e[[tr_name]]
    spl_tr  <- spl_e[[tr_name]]
    suffix  <- sub("H", "", tr_name)  # "01", "02", "12"
    cox_key <- tolower(tr_name)       # "h01", "h02", "h12"

    list(
      cox = list(
        time     = tr$cox$time,
        cumhaz   = tr$cox$cumhaz,
        haz_time = cox_tr$time,
        haz      = cox_tr$haz,
        ci_time  = tr$cox$ci_time,
        ci_low   = tr$cox$ci_low,
        ci_high  = tr$cox$ci_high
      ),
      weib = list(
        time    = tr$weib$time,
        cumhaz  = tr$weib$cumhaz,
        ci_time = tr$weib$ci_time,
        ci_low  = tr$weib$ci_low,
        ci_high = tr$weib$ci_high
      ),
      spl = list(
        time     = tr$spl$time,
        cumhaz   = tr$spl$cumhaz,
        haz_time = spl_tr$time,
        haz      = spl_tr$haz,
        ci_time  = tr$spl$ci_time,
        ci_low   = tr$spl$ci_low,
        ci_high  = tr$spl$ci_high
      ),
      idm_weib = list(
        times  = idm$idm_weib_times,
        cumhaz = idm[[paste0("idm_weib_", suffix)]]
      ),
      idm_splines = list(
        times  = idm$idm_splines_times,
        cumhaz = idm[[paste0("idm_splines_", suffix)]]
      ),
      cox_idm = list(
        time = cox_idm[[cox_key]]$time,
        haz  = cox_idm[[cox_key]]$haz
      )
    )
  }

  list(
    coefs = extract_coefs(fram_resfit_2026, epoch_name),
    H01   = build_transition("H01"),
    H02   = build_transition("H02"),
    H12   = build_transition("H12")
  )
}

# Build final list
result_list <- list(
  epoch1 = build_epoch("epoch1"),
  epoch2 = build_epoch("epoch2"),
  epoch3 = build_epoch("epoch3"),
  epoch4 = build_epoch("epoch4")
)

# Convert to JSON
json_output <- toJSON(result_list, pretty = TRUE, auto_unbox = TRUE, digits = NA)

# Write to file if needed
write(json_output, "/Users/jonas.arruda/PyCharm Projects/AmortizedSelectionBias/visit_censoring/baseline.json")

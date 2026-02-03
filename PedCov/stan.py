import logging
import numpy as np
from pathlib import Path

from cmdstanpy import CmdStanModel


BASE = Path(__file__).resolve().parent
stan_model = CmdStanModel(stan_file=BASE / 'stan' / 'pedcov_stan_model.stan')
stan_model_fixed_alpha = CmdStanModel(stan_file=BASE / 'stan' / 'pedcov_stan_model_fixed_alpha.stan')

def get_stan_posterior(obs_df, param_names, simulator, chains=4, show_progress=False, alpha=None,
                       iter_sampling=10000):
    """
    Prepare PedCov and fit the household infection model

    Parameters:
    obs_df: DataFrame
    max_t: maximum follow-up time
    chains: number of MCMC chains
    show_progress: whether to show progress during sampling
    alpha: baseline transmission rate (fixed model parameter)
    iter_sampling: number of sampling iterations (default is 10000)
    """
    # Ensure necessary columns are present
    required_columns = ['id_hh', 'id_patient', 'end_followup', 'date_sympt',
                        'age', 'infect_status', 'protected', 'first_test_pos', 'last_test_neg']
    if not all(column in obs_df.columns for column in required_columns):
        raise ValueError(f"DataFrame is missing one or more required columns: {required_columns}")
    if not simulator.variant in ['alpha', 'omicron']:
        raise ValueError(f"Simulator variant '{simulator.variant}' is not supported. Supported variants are 'alpha' and 'omicron'.")

    # Sort by household and patient ID for consistent indexing
    df_sorted = obs_df.sort_values(['id_hh', 'id_patient']).reset_index(drop=True)

    # Create mappings
    unique_hh = df_sorted['id_hh'].unique()
    hh_id_map = {hh_id: idx + 1 for idx, hh_id in enumerate(unique_hh)}
    hh_id_array = [hh_id_map[hh_id] for hh_id in df_sorted['id_hh']]

    # Get household sizes
    hh_sizes = df_sorted.groupby('id_hh')['id_patient'].count()
    hh_size_array = [hh_sizes[hh_id] for hh_id in unique_hh]

    # Convert other variables
    end_time = df_sorted['end_followup'].astype(int).tolist()
    obs_time = df_sorted['date_sympt'].astype(int).tolist()
    age_cat = df_sorted['age'].astype(int).tolist()
    infect_status = df_sorted['infect_status'].astype(int).tolist()
    protected = df_sorted['protected'].astype(int).tolist()
    last_test_neg = df_sorted['last_test_neg'].astype(int).tolist()
    first_test_pos = df_sorted['first_test_pos'].astype(int).tolist()

    # Get infected and susceptible indices (1-indexed for Stan)
    infected_mask = df_sorted['infect_status'] > 0
    susceptible_mask = df_sorted['infect_status'] == 0

    infected_idx = np.where(infected_mask)[0] + 1  # +1 for Stan indexing
    susceptible_idx = np.where(susceptible_mask)[0] + 1

    n_infected = len(infected_idx)
    n_susceptible = len(susceptible_idx)

    # Create household lookup tables
    household_ranges = {}
    pandas_to_sequential = {idx: i for i, idx in enumerate(df_sorted.index)}
    for hh_id in df_sorted['id_hh'].unique():
        hh_data = df_sorted[df_sorted['id_hh'] == hh_id]
        start_idx = hh_data.index[0]
        end_idx = hh_data.index[-1]
        infected_mask = hh_data['infect_status'] > 0

        if infected_mask.any():
            # Get infected individuals
            infected_data = hh_data[infected_mask]

            # Find who has the earliest symptom date
            earliest_idx = infected_data['date_sympt'].argmin()
            earliest_infected_pandas_idx = infected_data.index[earliest_idx]

            # Convert to relative position within household
            relative_pos = np.where(hh_data.index == earliest_infected_pandas_idx)[0][0]

            household_ranges[hh_id] = {
                'start': pandas_to_sequential[start_idx] + 1,  # 1-indexed for Stan
                'end': pandas_to_sequential[end_idx] + 1,
                'first_infected': relative_pos  # 0-indexed relative position
            }
        else:
            household_ranges[hh_id] = {
                'start': pandas_to_sequential[start_idx] + 1,
                'end': pandas_to_sequential[end_idx] + 1,
                'first_infected': 0  # Default when no infections
            }

    # Extract start and end indices for each household
    hh_start_idx = np.array([r['start'] for r in household_ranges.values()])
    hh_end_idx = np.array([r['end'] for r in household_ranges.values()])
    first_infected_idx = np.array([r['first_infected'] for r in household_ranges.values()])

    # Prepare Stan PedCov
    stan_data = {
        'N': len(df_sorted),
        'H': len(unique_hh),
        'hh_id': hh_id_array,
        'hh_size': hh_size_array,
        'end_time': end_time,
        'infect_status': infect_status,
        'obs_time': obs_time,
        'age_cat': age_cat,
        'is_protected': protected,
        'last_test_neg': last_test_neg,
        'first_test_pos': first_test_pos,

        'n_infected': n_infected,
        'n_susceptible': n_susceptible,
        'infected_idx': infected_idx.astype(int).tolist(),
        'susceptible_idx': susceptible_idx.astype(int).tolist(),
        'hh_start_idx': hh_start_idx.astype(int).tolist(),
        'hh_end_idx': hh_end_idx.astype(int).tolist(),
        'first_infected_idx': first_infected_idx.astype(int).tolist(),

        'kt_shape': simulator.shape_generation_time,  # generation time
        'kt_rate': 1/simulator.scale_generation_time,

        'inc_shape_symp': simulator.shapeIncub,  # incubation distribution
        'inc_rate_symp': 1/simulator.scaleIncub,
        'penalty_strength': 100,  # penalty strength for the latent incubation variables

        # fixed parameters
        #'mu_protect_acq': mu_protect_acq,
        #'mu_protect_transm': mu_protect_transm,
    }
    if alpha is not None:
        stan_data['alpha'] = alpha

    if show_progress:
        # Print PedCov summary for debugging
        logging.info(f"Data summary:")
        logging.info(f"  Number of households: {stan_data['H']}")
        logging.info(f"  Total individuals: {stan_data['N']}")
        logging.info(f"  Infections observed: {sum(inf > 0 for inf in infect_status)}")
        logging.info(f"  Protected individuals: {sum(protected)}")

    # Fit the model to the PedCov
    if alpha is None:
        stan_model_to_use = stan_model
    else:
        stan_model_to_use = stan_model_fixed_alpha
    fit = stan_model_to_use.sample(
        data=stan_data,
        show_progress=show_progress,
        chains=chains,
        iter_sampling=iter_sampling
    )

    # Extract posterior samples
    param_samples = {}
    for param in param_names:
        param_samples[param] = fit.draws_pd(param).values
    return param_samples

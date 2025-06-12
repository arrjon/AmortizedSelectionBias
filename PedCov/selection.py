import math
import numpy as np
import pandas as pd


def data_selection(d, variant, method):
    """
    Select households from simulation PedCov based on variant and method

    Parameters:
    -----------
    d : pd.DataFrame
        Household simulation PedCov
    variant : str
        Variant type ("alpha" or "omicron")
    method : str
        Selection method ("random", "pedcov", "adultcov")
    verbose : bool
        Whether to print debug messages

    Returns:
    --------
    pd.DataFrame : Selected and randomized household PedCov
    """
    d = d.copy()

    # Determine counts of households based on variant
    if variant == "alpha":
        tot_hh = 128  # 84
    elif variant == "omicron":
        tot_hh = 54   # 46
    else:
        raise ValueError("Variant must be 'alpha' or 'omicron'")

    tot_hh_a = math.ceil(tot_hh / 5)  # 1/5 of hh are included through asympto children
    tot_hh_s = tot_hh - tot_hh_a      # 4/5 of hh are included through sympto children

    if variant == "alpha":
        tot_hh_inclIndex = 88  # 49
    elif variant == "omicron":
        tot_hh_inclIndex = 44  # 36
    else:
        raise ValueError("Variant must be 'alpha' or 'omicron'")

    # Available households
    hh = list(d['id_hh'].unique())
    hh_origin = list(d['id_hh_origin'].unique())

    if method == "random":
        # Random sample from all households
        sel = np.random.choice(hh, size=min(tot_hh, len(hh)), replace=False)
        recruit = d[d['id_hh'].isin(sel)].copy()

    else:  # method in ["pedcov", "adultcov"]
        recruit = pd.DataFrame()

        hh_s = 0  # marker for count of sympto category
        hh_a = 0  # marker for count of asympto category
        hh_inclIndex = 0  # marker for count of inclusion=index category
        hh_inclNotIndex = 0  # marker for count of inclusion!=index category
        not_selected = 0

        while hh_s + hh_a < tot_hh and len(hh) > 0:
            # While there is not enough hh in the final base
            u = np.random.choice(hh)  # pick one randomly
            u_origin = d[d['id_hh'] == u]['id_hh_origin'].iloc[0]  # placeholder

            # Get household PedCov
            hh_data = d[d['id_hh'] == u]
            inclusion_case = hh_data[hh_data['is_incluCase'] == 1]

            if len(inclusion_case) == 0:
                # No inclusion case found, skip
                not_selected += 1
                hh = [h for h in hh if h != u]
                continue

            inclusion_case = inclusion_case.iloc[0]

            if method == "pedcov":
                # If inclusion case is an adult --> exclude and go to next iteration
                if inclusion_case['age_exact'] > 18:
                    not_selected += 1
                    hh = [h for h in hh if h != u]
                    continue
            elif method == "adultcov":
                # If inclusion case is NOT an adult --> exclude and go to next iteration
                if inclusion_case['age_exact'] <= 18:
                    not_selected += 1
                    hh = [h for h in hh if h != u]
                    continue

            # If inclusion case has not had symptoms or test yet --> exclude
            if inclusion_case['date_sympt'] >= inclusion_case['incl_dt']:
                not_selected += 1
                hh = [h for h in hh if h != u]  # Remove this hh from the list of pickable hh
                continue

            # Check if inclusion case is also index case
            is_index = inclusion_case['is_index'] == 1
            infect_status = inclusion_case['infect_status']

            if is_index and hh_inclIndex < tot_hh_inclIndex:
                # Inclusion case is also index case (and the count for this type is not full)
                if infect_status == 1 and hh_s < tot_hh_s:
                    # If inclusion case is symptomatic (and the count for this type is not full)
                    recruit = pd.concat([recruit, hh_data], ignore_index=True)
                    hh_s += 1  # increase count of sympto incl cases
                    hh_inclIndex += 1  # increase count of incl case also index

                    hh_origin = [h for h in hh_origin if h != u_origin]  # Remove from pickable hh_origin

                elif infect_status == 2 and hh_a < tot_hh_a:
                    # If inclusion case is asymptomatic (and the count for this type is not full)
                    recruit = pd.concat([recruit, hh_data], ignore_index=True)
                    hh_a += 1  # increase count of asympto incl cases
                    hh_inclIndex += 1  # increase count of incl case also index

                    hh_origin = [h for h in hh_origin if h != u_origin]  # Remove from pickable hh_origin

                else:
                    not_selected += 1  # If the count for this type of sympto status is already full

            elif not is_index and hh_inclNotIndex < (tot_hh - tot_hh_inclIndex):
                # Inclusion case is not the index case (and the count for this type is not full)
                if infect_status == 1 and hh_s < tot_hh_s:
                    # If inclusion case is symptomatic (and the count for this type is not full)
                    recruit = pd.concat([recruit, hh_data], ignore_index=True)
                    hh_s += 1  # increase count of sympto incl cases
                    hh_inclNotIndex += 1  # increase count of incl case not index

                    hh_origin = [h for h in hh_origin if h != u_origin]  # Remove from pickable hh_origin

                elif infect_status == 2 and hh_a < tot_hh_a:
                    # If inclusion case is asymptomatic (and the count for this type is not full)
                    recruit = pd.concat([recruit, hh_data], ignore_index=True)
                    hh_a += 1  # increase count of asympto incl cases
                    hh_inclNotIndex += 1  # increase count of incl case not index

                    hh_origin = [h for h in hh_origin if h != u_origin]  # Remove from pickable hh_origin

                else:
                    not_selected += 1  # If the count for this type of sympto status is already full

            else:
                not_selected += 1  # If the count for this type of incl case / index is already full

            hh = [h for h in hh if h != u]  # Remove this hh from the list of pickable hh
    # Randomize order of households
    if len(recruit) > 0:
        unique_households = recruit['id_hh'].unique()
        random_order = np.random.permutation(len(unique_households))

        # Create mapping from household ID to random order
        hh_order_map = dict(zip(unique_households, random_order))

        # Add random order column and sort
        recruit['random_order'] = recruit['id_hh'].map(hh_order_map)
        randomized_recruit = recruit.sort_values('random_order').drop('random_order', axis=1).reset_index(drop=True)
    else:
        randomized_recruit = recruit

    return randomized_recruit

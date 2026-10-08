import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from bp_generation import (
    generate_hypertension_stage,
    generate_measurement_frequency,
    generate_new_DBP_value,
    generate_new_SBP_value,
    generate_treatment_prob,
)
from mortality import (
    add_insurance_mortality,
    create_death_adjustment_table,
    load_all_life_tables,
    NHB_non_insurance_prop,
    NHW_non_insurance_prop,
)
from numeric import convert_to_prob, convert_to_rate

# parent directory
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
overall_folder = os.path.dirname(parent_directory)

# Load all 8 (race, sex_ind) life tables, then apply insurance adjustment to
# the four NHB/NHW entries used by the simulation. The H/O entries stay in the
# dict but are unused here.
life_table_dict = load_all_life_tables(overall_folder)
for race, prop in [("NHB", NHB_non_insurance_prop), ("NHW", NHW_non_insurance_prop)]:
    for sex_ind in (0, 1):
        life_table_dict[(race, sex_ind)] = add_insurance_mortality(
            life_table_dict[(race, sex_ind)], prop
        )

CYCLE_LENGTH = 1 / 12.0

##probs for out of health system to into health system
pOI_ins = 0.0025
# rrOI_no_ins = 0.6
# pOI_no_ins = convert_to_prob(convert_to_rate(pOI_ins) * rrOI_no_ins)

##probs for in health system to detected/treated
pDT_ins = 0.50
# rrDT_no_ins = 0.5
# pDT_no_ins = convert_to_prob(convert_to_rate(pDT_ins) * rrDT_no_ins)

##probs for discontinuing treatment
# ##ALLHAT
##fraction adhered in year 1 to TD or same class: 0.871 (blinded drug: 0.839)
##fraction adhered in year 5 to TD or same class: 0.805 (blinded drug: 0.712)
TD_adherence_year_1 = 0.839  # 0.871
TD_adherence_year_5 = 0.712  # 0.805
TD_disc_rate = 1 - (TD_adherence_year_5 / TD_adherence_year_1) ** (1 / 4)
pDTUT_TD = convert_to_prob(TD_disc_rate * CYCLE_LENGTH)

##fraction adhered in year 1 to ACE or same class: 0.824 (blinded drug: 0.774)
##fraction adhered in year 5 to ACE or same class: 0.726 (blinded drug: 0.612)
ACE_adherence_year_1 = 0.774  # 0.824
ACE_adherence_year_5 = 0.612  # 0.726
ACE_disc_rate = 1 - (ACE_adherence_year_5 / ACE_adherence_year_1) ** (1 / 4)
pDTUT_ACE = convert_to_prob(ACE_disc_rate * CYCLE_LENGTH)

##There is no difference by race for TD
##for ACE wright et. al did observe differences
ACE_adherence_year_1_black = 0.76
ACE_adherence_year_5_black = 0.57
ACE_adherence_year_1_nonblack = 0.78
ACE_adherence_year_5_nonblack = 0.63
ACE_disc_rate_black = 1 - (ACE_adherence_year_5_black / ACE_adherence_year_1_black) ** (
    1 / 4
)
pDTUT_ACE_black = convert_to_prob(ACE_disc_rate_black * CYCLE_LENGTH)
ACE_disc_rate_nonblack = 1 - (
    ACE_adherence_year_5_nonblack / ACE_adherence_year_1_nonblack
) ** (1 / 4)
pDTUT_ACE_nonblack = convert_to_prob(ACE_disc_rate_nonblack * CYCLE_LENGTH)

# Increase risk of discontinuation with no insurance
# rrDTUT_no_ins = 5


def generate_transitions_HS(
    current_state_HS,
    current_state_BP,
    age,
    sex,
    systolic_bp,
    diastolic_bp,
    treatment,
    measurement_interval,
    measurement_quantile,
    hetero_effect,
    race,
):
    transition_vec = {
        "OHS": [0, 0, 0, 0],
        "IHS": [0, 0, 0, 0],
        "DT": [0, 0, 0, 0],
        "DUT": [0, 0, 0, 0],
    }
    ##we assume everyone has insurance
    # this prob does not matter since everyone has insurance already
    pOI = pOI_ins

    # Transition vector for current_state_DNH "D"
    if current_state_BP == "D":
        transition_vec["OHS"] = [1, 0, 0, 0]
        transition_vec["IHS"] = [0, 1, 0, 0]
        transition_vec["DT"] = [0, 0, 1, 0]
        transition_vec["DUT"] = [0, 0, 0, 1]
    else:
        if measurement_interval == 0:
            pDT = generate_treatment_prob(
                logistic_treatment_model,
                age,
                sex,
                systolic_bp,
                diastolic_bp,
            )
            measurement_interval = int(
                generate_measurement_frequency(
                    measurement_frequency_models,
                    quantiles.index(measurement_quantile),
                    age,
                    sex,
                    systolic_bp,
                    diastolic_bp,
                )
            )
        else:
            pDT = 0
            measurement_interval = measurement_interval - 1

        if treatment == "ACE":
            # if hetero_effect = True, we assume heterogenous discontinuation rates for ACE
            if hetero_effect:
                if race == "NHB":
                    pDTUT = pDTUT_ACE_black
                else:
                    pDTUT = pDTUT_ACE_nonblack
            else:
                pDTUT = pDTUT_ACE
        else:
            pDTUT = pDTUT_TD

        # Transition probabilities for "S1" or "S2"
        if current_state_BP in ["S1", "S2"]:
            transition_vec["OHS"] = [1 - pOI, pOI, 0, 0]
            transition_vec["IHS"] = [0, 1 - pDT, pDT, 0]
        else:
            transition_vec["OHS"] = [1 - pOI, pOI, 0, 0]
            # We don't allow people to transition to treatment until stage 1/2
            transition_vec["IHS"] = [0, 1, 0, 0]

        transition_vec["DT"] = [0, 0, 1 - pDTUT, pDTUT]
        if current_state_BP in ["S1", "S2"]:
            transition_vec["DUT"] = [0, 0, pDT, 1 - pDT]
        else:
            transition_vec["DUT"] = [0, 0, 0, 1]

    return transition_vec[current_state_HS], measurement_interval


quantiles = np.arange(0.01, 1.0, 0.01)
quantiles = [round(x, 2) for x in quantiles]

sb_quantile_models = [
    sm.load(
        f"{overall_folder}/data_and_models/afc_models/standard/starting_systolic_bp_regressions/SBP_quantile_{q}_model.pickle"
    )
    for q in quantiles
]
db_quantile_models = [
    sm.load(
        f"{overall_folder}/data_and_models/afc_models/standard/starting_diastolic_bp_regressions/DBP_quantile_{q}_model.pickle"
    )
    for q in quantiles
]

systolic_splines = pd.read_csv(
    f"{overall_folder}/data_and_models/afc_outputs/standard/systolic_splines.csv"
)
diastolic_splines = pd.read_csv(
    f"{overall_folder}/data_and_models/afc_outputs/standard/diastolic_splines.csv"
)

measurement_frequency_models = [
    sm.load(
        f"{overall_folder}/data_and_models/afc_models/standard/measurement_models_pre_treatment/measure_freq_quantile_{q}_model.pickle"
    )
    for q in quantiles
]

logistic_treatment_model = sm.load(
    f"{overall_folder}/data_and_models/afc_models/standard/treatment_models/logistic_treatment_model.pickle"
)


# These represent ALLHAT trial systolic and diastolic blood pressure value drops from being on TD (thiazide diuretic) or
# ACE (ace inhibitors).
# When we assume heterogeneous treatment effects, we use differential effects for Black vs. non-Black patients
# When we assume no heterogenerous treatment effects, we use the overall effect across the patient population
treatment_effects = {
    "TD_black": {
        1: [7.7, 3.9],
        2: [8.6, 5.0],
        4: [10.5, 6.6],
    },
    "ACE_black": {
        1: [2.5, 2.3],
        2: [3.4, 3.4],
        4: [6.8, 5.6],
    },
    "TD_non_black": {
        1: [9.8, 4.7],
        2: [10.6, 5.8],
        4: [12.3, 7.6],
    },
    "ACE_non_black": {
        1: [8.1, 4.9],
        2: [9.5, 6.1],
        4: [12.0, 8.0],
    },
    "TD_overall": {
        1: [9.3, 4.7],
        2: [10.3, 5.7],
        4: [12.3, 7.5],
    },
    "ACE_overall": {
        1: [6.1, 4.2],
        2: [8.0, 5.5],
        4: [10.9, 7.5],
    },
}


def get_treatment_effect(race, treatment, years_treated, hetero_effect):
    if hetero_effect:
        group = f"{treatment}_{'black' if race == 'NHB' else 'non_black'}"
    else:
        group = f"{treatment}_overall"

    if years_treated < 1:
        return treatment_effects[group][1]
    elif years_treated < 2:
        return treatment_effects[group][2]
    else:
        return treatment_effects[group][4]


def generate_next_BP_stage(
    this_quantile,
    this_quantile_dbp,
    current_state_HS,
    time_treated,
    current_state_BP,
    age,
    sex,
    race,
    treatment,
    hetero_effect,
):
    if current_state_BP != "D":
        new_SBP = generate_new_SBP_value(
            sb_quantile_models,
            systolic_splines,
            this_quantile,
            age,
            sex,
        )
        new_DBP = generate_new_DBP_value(
            db_quantile_models,
            diastolic_splines,
            this_quantile_dbp,
            age,
            sex,
        )

        if current_state_HS == "DT":
            years_treated = time_treated * CYCLE_LENGTH
            effect = get_treatment_effect(race, treatment, years_treated, hetero_effect)
            new_SBP = new_SBP - effect[0]
            new_DBP = new_DBP - effect[1]

        return new_SBP, new_DBP, generate_hypertension_stage(new_SBP, new_DBP)
    else:
        return np.nan, np.nan, "D"


DBP_values = [65, 72, 76, 82, 90]
DBP_HR_values = [1, 1.8, 2, 2, 3]
SBP_values = [115, 124, 130, 140, 155]
SBP_HR_values = [1, 2, 2, 2.4, 4]

hr_adjust_table = create_death_adjustment_table()


def generate_transitions_DNH(
    current_state_BP,
    current_DBP_val,
    current_SBP_val,
    age,
    sex,
    race,
):
    if current_state_BP != "D":
        current_state_DNH = "A"
    else:
        current_state_DNH = "D"

    transition_vec = {"A": [0, 0], "D": [0, 1]}  # Default transition for age >= 100

    if age >= 100:
        transition_vec["A"] = [0, 1]
        return transition_vec[current_state_DNH]

    # Select the appropriate life table based on race and sex
    this_life_table = life_table_dict.get((race, sex))

    pHD_column = "qx_cycle"
    pHD = this_life_table[this_life_table["age"] == int(age)][pHD_column].iloc[0]

    rHD = convert_to_rate(pHD)
    # Calculate hazard ratios
    DBP_HR = np.interp(current_DBP_val, DBP_values, DBP_HR_values, left=1, right=3)
    SBP_HR = np.interp(current_SBP_val, SBP_values, SBP_HR_values, left=1, right=4)
    avg_HR = np.mean([DBP_HR, SBP_HR])

    HR_adjustment = hr_adjust_table[hr_adjust_table["age"] == int(age)]["avg_HR"].iloc[
        0
    ]

    # rSD = rHD * avg_HR
    rSD = (rHD * avg_HR) / HR_adjustment

    pSD = convert_to_prob(rSD)

    transition_vec["A"] = [1 - pSD, pSD]
    return transition_vec[current_state_DNH]


def sample_quantile_change_schedule(
    N,
    cycles,
    random_seeds,
    mean_years_between_changes=5,
    min_gap_months=12,
):
    mean_months = mean_years_between_changes * 12

    schedules = []

    for i in range(N):
        random_seed = random_seeds.iloc[i]
        rng = np.random.default_rng(random_seed)
        t = 0
        person_schedule = []

        while True:
            gap = int(round(rng.exponential(mean_months)))
            gap = max(gap, min_gap_months)  # avoid unrealistically frequent changes
            t += gap

            if t > cycles:
                break

            person_schedule.append(t)

        schedules.append(person_schedule)

    return schedules

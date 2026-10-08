import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from numeric import convert_to_cycle, convert_to_prob, convert_to_rate
from paths import overall_folder

# Hazard ratio for increased mortality risk among the uninsured.
# Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC2775760/
HAZARD_RATIO = 1.4
# Racial/ethnic group-specific prevalence of uninsured individuals.
# Source: https://www.kff.org/racial-equity-and-health-policy/issue-brief/health-coverage-by-race-and-ethnicity/
NHW_non_insurance_prop = 0.066
NHB_non_insurance_prop = 0.10

LIFE_TABLE_FILES = {
    ("NHB", 0): "NonHispanicBlackMale.xlsx",
    ("NHB", 1): "NonHispanicBlackFemale.xlsx",
    ("NHW", 0): "NonHispanicWhiteMale.xlsx",
    ("NHW", 1): "NonHispanicWhiteFemale.xlsx",
    ("H", 0): "HispanicMale.xlsx",
    ("H", 1): "HispanicFemale.xlsx",
    ("O", 0): "OverallMale.xlsx",
    ("O", 1): "OverallFemale.xlsx",
}


def transform_lifetables(life_table):
    life_table = life_table[:101]
    life_table = life_table.reset_index()
    life_table = life_table.drop(columns=["index"])
    life_table["age"] = pd.Series(list(range(0, 101)), index=life_table.index)
    life_table["qx_cycle"] = convert_to_cycle(life_table["qx"])
    return life_table


def load_all_life_tables(overall_folder):
    base = f"{overall_folder}/data_and_models/lifetables_2021"
    return {
        key: transform_lifetables(
            pd.read_excel(f"{base}/{filename}", engine="openpyxl")
        )
        for key, filename in LIFE_TABLE_FILES.items()
    }


def add_insurance_mortality(lifetable, non_insurance_prop):
    lifetable = transform_lifetables(lifetable)

    p_insured = 1 - non_insurance_prop
    insured_probs = [0 for i in range(len(lifetable))]
    notinsured_probs = [0 for i in range(len(lifetable))]
    for i in range(len(lifetable)):
        if lifetable["age"].iloc[i] != 100:
            this_rate = convert_to_rate(lifetable["qx"].iloc[i])
            insured_probs[i] = convert_to_prob(
                this_rate / (p_insured + HAZARD_RATIO * (1 - p_insured))
            )
            notinsured_probs[i] = convert_to_prob(
                convert_to_rate(insured_probs[i]) * HAZARD_RATIO
            )
        else:
            insured_probs[i] = 1
            notinsured_probs[i] = 1
    lifetable["qx_ins"] = pd.Series(insured_probs, index=lifetable.index)
    lifetable["qx_no_ins"] = pd.Series(notinsured_probs, index=lifetable.index)
    lifetable["qx_cycle"] = convert_to_cycle(lifetable["qx"])
    lifetable["qx_ins_cycle"] = convert_to_cycle(lifetable["qx_ins"])
    lifetable["qx_no_ins_cycle"] = convert_to_cycle(lifetable["qx_no_ins"])
    return lifetable


def return_estimated_HR(sbp_val, dbp_val):
    DBP_values = [65, 72, 76, 82, 90]
    DBP_HR_values = [1, 1.8, 2, 2, 3]
    SBP_values = [115, 124, 130, 140, 155]
    SBP_HR_values = [1, 2, 2, 2.4, 4]

    DBP_HR = np.interp(dbp_val, DBP_values, DBP_HR_values, left=1, right=3)
    SBP_HR = np.interp(sbp_val, SBP_values, SBP_HR_values, left=1, right=4)
    return np.mean([DBP_HR, SBP_HR])


def create_death_adjustment_table():
    avg_sbp_dbp = pd.read_csv(
        f"{overall_folder}/data_and_models/nhanes_inputs/bp_means_by_age_5yr.csv"
    )
    avg_sbp_dbp["age_group_start"] = (
        avg_sbp_dbp["age_group"].str.extract(r"^(\d+)").astype(int)
    )

    f_sbp = interp1d(
        avg_sbp_dbp["age_group_start"],
        avg_sbp_dbp["mean_sbp"],
        bounds_error=False,
        fill_value=(avg_sbp_dbp["mean_sbp"].iloc[0], avg_sbp_dbp["mean_sbp"].iloc[-1]),
    )
    f_dbp = interp1d(
        avg_sbp_dbp["age_group_start"],
        avg_sbp_dbp["mean_dbp"],
        kind="linear",
        bounds_error=False,
        fill_value=(avg_sbp_dbp["mean_dbp"].iloc[0], avg_sbp_dbp["mean_dbp"].iloc[-1]),
    )
    sbp_val = []
    dbp_val = []
    avg_hr = []
    for a in range(40, 100):
        sbp_val.append(f_sbp(a))
        dbp_val.append(f_dbp(a))
        avg_hr.append(return_estimated_HR(f_sbp(a), f_dbp(a)))

    df = pd.DataFrame(list(range(40, 100)), columns=["age"])
    df["sbp_mean"] = pd.Series(sbp_val)
    df["dbp_mean"] = pd.Series(dbp_val)
    df["avg_HR"] = pd.Series(avg_hr)
    return df

import os
import pickle

import numpy as np
import pandas as pd

from paths import overall_folder
from results_misc import cycles
from stats import compute_mean_sd


def create_cohort_demographics(sampled_cohort, folder, social, overall=True):
    demo_dict = {}
    demo_dict["insurance"] = compute_mean_sd(sampled_cohort["insurance"])
    demo_dict["SDI"] = compute_mean_sd(sampled_cohort["SDI"])

    demo_dict["systolic_bp"] = compute_mean_sd(sampled_cohort["predicted_systolic_bp"])

    demo_dict["diastolic_bp"] = compute_mean_sd(
        sampled_cohort["predicted_diastolic_bp"]
    )

    demo_dict["month_difference"] = compute_mean_sd(
        sampled_cohort["predicted_month_difference"]
    )

    demo_dict["female"] = compute_mean_sd(sampled_cohort["female"])

    demo_dict["month_difference"] = compute_mean_sd(sampled_cohort["month_difference"])

    demo_dict["visits"] = compute_mean_sd(sampled_cohort["visits"])

    hyp_stages = pd.get_dummies(sampled_cohort["predicted_BP_stage"], dtype=int)
    demo_dict["Normal"] = compute_mean_sd(hyp_stages["N"])
    demo_dict["Elevated"] = compute_mean_sd(hyp_stages["E"])
    demo_dict["Stage 1"] = compute_mean_sd(hyp_stages["S1"])
    demo_dict["Stage 2"] = compute_mean_sd(hyp_stages["S2"])

    demo_dict["NHB"] = compute_mean_sd(sampled_cohort["black"])
    demo_dict["place"] = compute_mean_sd(sampled_cohort["place"])

    stats_dir = f"{overall_folder}/results/{folder}/cohort_statistics/{'social' if social else 'standard'}"
    os.makedirs(stats_dir, exist_ok=True)

    if overall:
        with open(f"{stats_dir}/demo.pkl", "wb") as f:
            pickle.dump(demo_dict, f)

    return demo_dict


def create_cohort_demographics_by_race(sampled_cohort, folder, social):
    demo_dict_NHB = create_cohort_demographics(
        sampled_cohort[sampled_cohort["black"] == 1], folder, social, overall=False
    )

    demo_dict_NHW = create_cohort_demographics(
        sampled_cohort[sampled_cohort["black"] == 0], folder, social, overall=False
    )

    stats_dir = f"{overall_folder}/results/{folder}/cohort_statistics/{'social' if social else 'standard'}"
    os.makedirs(stats_dir, exist_ok=True)

    with open(f"{stats_dir}/demo_NHB.pkl", "wb") as f:
        pickle.dump(demo_dict_NHB, f)

    with open(f"{stats_dir}/demo_NHW.pkl", "wb") as f:
        pickle.dump(demo_dict_NHW, f)

    return demo_dict_NHB, demo_dict_NHW


def compute_weighted_htn_prevalence_by_age(
    trace_df,
    start_age=40,
    max_cycle=cycles,
    hs_prefix="HS-Year ",
    bp_prefix="BPStageYear ",
):
    monthly_results = []

    for c in range(max_cycle + 1):
        hs_col = f"{hs_prefix}{c}"
        bp_col = f"{bp_prefix}{c}"

        hs = trace_df[hs_col]
        bp = trace_df[bp_col]

        alive = bp != "D"
        htn = alive & (bp.isin(["S1", "S2"]) | (hs == "DT"))

        n_alive = alive.sum()
        n_htn = htn.sum()
        prevalence = n_htn / n_alive if n_alive > 0 else np.nan

        age_exact = start_age + c / 12.0
        age_int = int(np.floor(age_exact))

        monthly_results.append(
            {
                "cycle": c,
                "age_exact": age_exact,
                "age": age_int,
                "n_alive": n_alive,
                "n_htn": n_htn,
                "prevalence": prevalence,
            }
        )

    monthly_df = pd.DataFrame(monthly_results)

    age_prev_df = monthly_df.groupby("age", as_index=False).agg(
        prevalence=("prevalence", "mean"),
        mean_n_alive=("n_alive", "mean"),
        mean_n_htn=("n_htn", "mean"),
        n_months=("cycle", "count"),
    )

    age_prev_df = age_prev_df[
        (age_prev_df["age"] >= 40) & (age_prev_df["age"] <= 100)
    ].copy()

    return age_prev_df

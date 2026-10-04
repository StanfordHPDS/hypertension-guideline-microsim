import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROUND_DIGITS = 1


def combine_se_errors(*ses):
    return np.sqrt(sum(s**2 for s in ses))


def compute_mean_sd(series):
    mean = round(series.mean(), ROUND_DIGITS)
    mcse = round(series.std(), ROUND_DIGITS)
    return [mean, mcse]


def compute_mean_mcse(series):
    mean = series.mean()
    mcse = series.std() / np.sqrt(len(series))
    return [mean, mcse]


def compute_inc_mean_sd(series):
    mean = series.mean()
    mcse = np.sqrt((mean * (1 - mean)) / len(series))
    return [mean, mcse]


def compute_difference_mean_se(mean1, mean2, se1, se2):
    mean = mean1 - mean2
    mcse = combine_se_errors(se1, se2)
    return [mean, mcse]


def create_overall_results(total_trace):
    results_dict = dict()

    results_dict["cum_inc_sick"] = compute_inc_mean_sd(total_trace["was_sick"])
    results_dict["years_sick"] = compute_mean_mcse(
        total_trace[total_trace["was_sick"] == 1]["years_sick"]
    )
    results_dict["sick_age"] = compute_mean_mcse(
        total_trace[total_trace["was_sick"] == 1]["sick_age"]
    )
    results_dict["years_sick_treated"] = compute_mean_mcse(
        total_trace[total_trace["was_treated"] == 1]["years_sick_treated"]
    )
    results_dict["years_to_treated"] = compute_mean_mcse(
        total_trace[total_trace["was_treated"] == 1]["treated_age"]
        - total_trace[total_trace["was_treated"] == 1]["sick_age"]
    )
    results_dict["years_healthy_treated"] = compute_mean_mcse(
        total_trace[total_trace["was_treated"] == 1]["years_treated"]
        - total_trace[total_trace["was_treated"] == 1]["years_sick_treated"]
    )
    results_dict["years_treated"] = compute_mean_mcse(
        total_trace[total_trace["was_treated"] == 1]["years_treated"]
    )
    results_dict["treated_age"] = compute_mean_mcse(
        total_trace[total_trace["was_treated"] == 1]["treated_age"]
    )
    x = total_trace[total_trace["was_treated"] == 1]["years_treated"]
    results_dict["uncontrolled_percent"] = compute_inc_mean_sd(
        total_trace[total_trace["was_treated"] == 1]["years_sick_treated"] / x
    )
    results_dict["cum_inc_treated"] = compute_inc_mean_sd(total_trace["was_treated"])
    TD_dummy = pd.get_dummies(total_trace["treatment_type"], dtype=int)["TD"]
    results_dict["cum_inc_treated_TD"] = compute_inc_mean_sd(
        TD_dummy * total_trace["was_treated"]
    )
    results_dict["years_to_death"] = compute_mean_mcse(total_trace["years_to_death"])

    return results_dict


def create_results_by_group(total_trace, group_type):
    if group_type == "race":
        groups = ["NHW", "NHB"]
        column = "race"

    if group_type == "insurance":
        column = "insurance"
        groups = [1, 0]

    group_dict = dict()
    for g in groups:
        total_trace_g = total_trace[total_trace[column] == g]
        group_dict[g] = create_overall_results(total_trace_g)

    diff_dict = dict()

    for k in group_dict[groups[0]].keys():
        mean_1, sd_1 = group_dict[groups[0]][k]
        mean_2, sd_2 = group_dict[groups[1]][k]
        diff_dict[k] = compute_difference_mean_se(mean_1, mean_2, sd_1, sd_2)

    group_dict["Difference"] = diff_dict
    return group_dict


def create_results_multiple_groups(total_trace, group_type):
    if group_type == "insurance":
        column = "insurance"
        groups = [1, 0]

    group_dict = dict()
    for r in ["NHB", "NHW"]:
        for g in groups:
            total_trace_g = total_trace[
                (total_trace["race"] == r) & (total_trace[column] == g)
            ]
            group_dict[(g, r)] = create_overall_results(total_trace_g)

    return group_dict


def compute_km_cumulative_incidence_by_group(
    df,
    starting_age=40,
    sex_col="sex",
    race_col="race",
    female_value=1,
):
    df = df.copy()

    df["death_age"] = df["years_to_death"] + starting_age

    df["event_or_censor_age"] = np.where(
        df["was_sick"] == 1, df["sick_age"], df["death_age"]
    )

    df["group"] = np.where(
        df[sex_col] == female_value, df[race_col] + " female", df[race_col] + " male"
    )

    group_order = ["Overall", "NHW male", "NHW female", "NHB male", "NHB female"]

    results = []
    km = KaplanMeierFitter()
    km.fit(
        durations=df["event_or_censor_age"],
        event_observed=df["was_sick"],
        label="Overall",
    )

    overall_ci = 1 - km.survival_function_
    overall_ci = overall_ci.reset_index()
    overall_ci.columns = ["age", "cumulative_incidence"]
    overall_ci["group"] = "Overall"
    results.append(overall_ci)

    for group in ["NHW male", "NHW female", "NHB male", "NHB female"]:
        sub = df[df["group"] == group].copy()

        if len(sub) == 0:
            continue

        km = KaplanMeierFitter()
        km.fit(
            durations=sub["event_or_censor_age"],
            event_observed=sub["was_sick"],
            label=group,
        )

        ci = 1 - km.survival_function_
        ci = ci.reset_index()
        ci.columns = ["age", "cumulative_incidence"]
        ci["group"] = group
        results.append(ci)

    out_df = pd.concat(results, ignore_index=True)
    out_df["group"] = pd.Categorical(
        out_df["group"], categories=group_order, ordered=True
    )
    out_df = out_df.sort_values(["group", "age"]).reset_index(drop=True)

    return out_df

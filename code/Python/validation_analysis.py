import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from bq_config import bq_client, table
import json
import dask.dataframe as dd
import statsmodels.stats.proportion
from argparse import ArgumentParser
from lifelines import KaplanMeierFitter
from results_misc import simulate_death_age


def conduct_KS_test_measurement_intervals(overall_folder, folder, cohort):
    client = bq_client()

    query = f"""
    SELECT * from {table("SBP_DBP_values_ordered_nonmeds_all")} sl
    where SDI is not NULL and month_difference < 25 and age = 40
    ORDER BY RAND()
    LIMIT 10000
    """
    query_job = client.query(query)
    df = query_job.to_dataframe()

    plt.hist(df["month_difference"])
    plt.xlabel("Months")
    plt.savefig(
        f"{overall_folder}/figures/supplement/afc_subset_month_difference.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.hist(cohort["predicted_month_difference"])
    plt.xlabel("Months")
    plt.savefig(
        f"{overall_folder}/figures/supplement/cohort_predicted_month_difference.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    os.makedirs(f"results/{folder}/cohort_statistics/standard", exist_ok=True)

    with open(
        f"results/{folder}/cohort_statistics/standard/month_ks_value.json", "w"
    ) as json_file:
        ks_value = stats.kstest(
            cohort["predicted_month_difference"], df["month_difference"]
        )[0]
        json.dump(ks_value, json_file)
    with open(
        f"results/{folder}/cohort_statistics/standard/month_ks_p_value.json", "w"
    ) as json_file:
        ks_p_value = stats.kstest(
            cohort["predicted_month_difference"], df["month_difference"]
        )[1]
        json.dump(ks_p_value, json_file)


def load_and_concat_parquet(prefix, overall_folder, folder):
    dfs = [
        dd.read_parquet(
            f"{overall_folder}/validation_results/{folder}/saved_results/{prefix}_{suffix}/"
        )
        for suffix in ["1", "2"]
    ]
    return dd.concat(dfs, axis=0)


def read_in_afc_validation_files(overall_folder, folder):
    BP_state_trace_df = load_and_concat_parquet("BP_state", overall_folder, folder)
    statistics_df = load_and_concat_parquet("statistics", overall_folder, folder)
    sampled_cohort = pd.read_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_cohort.csv"
    )
    N = len(sampled_cohort)

    race_group_conditions = [
        (sampled_cohort["black"] == 1),
        (sampled_cohort["white"] == 1),
        (sampled_cohort["hispanic"] == 1),
    ]

    race_options = ["NHB", "NHW", "H"]
    race_values = np.select(race_group_conditions, race_options, default="O")

    population_df = pd.DataFrame(
        {
            "id": range(N),
            "starting_age": [40] * N,
            "race": race_values,
            "sex": sampled_cohort["female"],
            "insurance": ["Y"] * N,
            "SDI": sampled_cohort["SDI"],
        }
    )
    total_trace = (
        dd.concat([BP_state_trace_df, statistics_df], axis=1).compute().reset_index()
    )
    total_trace = pd.concat([population_df, total_trace], axis=1)

    return total_trace, sampled_cohort


def create_afc_validation_summary_stats(sampled_cohort):
    black_prop = sampled_cohort["black"].mean()
    white_prop = sampled_cohort["white"].mean()
    hispanic_prop = sampled_cohort["hispanic"].mean()
    female_prop = sampled_cohort["female"].mean()
    BP_stage_proportions = sampled_cohort["predicted_BP_stage"].value_counts(
        normalize=True
    )
    sdi_mean = sampled_cohort["SDI"].mean()
    sdi_std = sampled_cohort["SDI"].std()
    systolic_mean = sampled_cohort["predicted_systolic_bp"].mean()
    systolic_std = sampled_cohort["predicted_systolic_bp"].std()
    diastolic_mean = sampled_cohort["predicted_diastolic_bp"].mean()
    diastolic_std = sampled_cohort["predicted_diastolic_bp"].std()

    with open("results/afc_validation/black_prop.json", "w") as json_file:
        json.dump(black_prop, json_file)
    with open("results/afc_validation/white_prop.json", "w") as json_file:
        json.dump(white_prop, json_file)
    with open("results/afc_validation/hispanic_prop.json", "w") as json_file:
        json.dump(hispanic_prop, json_file)
    with open("results/afc_validation/female_prop.json", "w") as json_file:
        json.dump(female_prop, json_file)
    for stage in ["N", "E", "S1", "S2"]:
        with open(f"results/afc_validation/{stage}_prop.json", "w") as json_file:
            json.dump(BP_stage_proportions.get(stage, 0.0), json_file)
    with open("results/afc_validation/sdi_mean.json", "w") as json_file:
        json.dump(sdi_mean, json_file)
    with open("results/afc_validation/sdi_std.json", "w") as json_file:
        json.dump(sdi_std, json_file)
    with open("results/afc_validation/systolic_mean.json", "w") as json_file:
        json.dump(systolic_mean, json_file)
    with open("results/afc_validation/systolic_std.json", "w") as json_file:
        json.dump(systolic_std, json_file)
    with open("results/afc_validation/diastolic_mean.json", "w") as json_file:
        json.dump(diastolic_mean, json_file)
    with open("results/afc_validation/diastolic_std.json", "w") as json_file:
        json.dump(diastolic_std, json_file)


def create_trace_arrs(total_trace, cycles):
    BP_state_trace_df_columns = ["BPStageYear " + str(x) for x in range(0, cycles + 1)]
    N = len(total_trace)
    N_arr = []
    E_arr = []
    S1_arr = []
    S2_arr = []
    for i in BP_state_trace_df_columns[:-120]:
        if N - len(np.where(total_trace[i] == "D")[0]) != 0:
            N_arr.append(
                len(np.where(total_trace[i] == "N")[0])
                / (N - len(np.where(total_trace[i] == "D")[0]))
            )
            E_arr.append(
                len(np.where(total_trace[i] == "E")[0])
                / (N - len(np.where(total_trace[i] == "D")[0]))
            )
            S1_arr.append(
                len(np.where(total_trace[i] == "S1")[0])
                / (N - len(np.where(total_trace[i] == "D")[0]))
            )
            S2_arr.append(
                len(np.where(total_trace[i] == "S2")[0])
                / (N - len(np.where(total_trace[i] == "D")[0]))
            )

    return N_arr, E_arr, S1_arr, S2_arr


def create_validation_outputs(total_trace, cycles, N_arr, E_arr, S1_arr, S2_arr):
    BP_state_trace_df_columns = ["BPStageYear " + str(x) for x in range(0, cycles + 1)]
    AFC_prev = pd.read_csv("data_and_models/afc_outputs/standard/AFC_BP_Stage_Prev.csv")
    N = len(total_trace)

    x = list(range(0, cycles + 1))
    age_range = [40 + i / 12 for i in x]
    stage_names = ["N", "E", "S1", "S2"]
    stage_arr = [N_arr, E_arr, S1_arr, S2_arr]
    stage_names2 = ["Normal", "Elevated", "Stage 1", "Stage 2"]
    for s in range(len(stage_names)):
        stage_lower = []
        stage_upper = []
        for i in BP_state_trace_df_columns[:-120]:
            num1 = len(np.where(total_trace[i] == stage_names[s])[0])
            num2 = N - len(np.where(total_trace[i] == "D")[0])
            stage_lower.append(
                statsmodels.stats.proportion.proportion_confint(
                    num1, num2, alpha=0.05, method="normal"
                )[0]
            )
            stage_upper.append(
                statsmodels.stats.proportion.proportion_confint(
                    num1, num2, alpha=0.05, method="normal"
                )[1]
            )
        plt.figure(figsize=(8, 5))
        plt.plot(
            age_range[: len(stage_arr[s])],
            stage_arr[s],
            color="Blue",
            label="Simulation cohort",
        )
        plt.fill_between(
            age_range[: len(stage_arr[s])],
            (stage_lower),
            (stage_upper),
            color="blue",
            alpha=0.1,
        )
        plt.plot(
            AFC_prev["age"],
            AFC_prev[stage_names2[s]],
            color="Red",
            label="PRIME Registry",
        )
        plt.fill_between(
            AFC_prev["age"],
            AFC_prev[stage_names2[s] + " 2.5%"],
            AFC_prev[stage_names2[s] + " 97.5%"],
            color="red",
            alpha=0.1,
        )
        plt.legend()
        plt.xlabel("Age")
        plt.ylabel("Stage prevalence")
        plt.savefig(
            f"figures/supplement/validation_{stage_names2[s]}_trace.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def export_validation_death_sick_statistics(total_trace):
    with open("results/afc_validation/death_age_mean.json", "w") as json_file:
        json.dump(total_trace["years_to_death"].mean(), json_file)
    with open("results/afc_validation/death_age_se.json", "w") as json_file:
        json.dump(
            total_trace["years_to_death"].std() / np.sqrt(len(total_trace)), json_file
        )

    race_groups = total_trace["race"].unique()
    for r in race_groups:
        print(r)
        print(total_trace[total_trace["race"] == r]["death_age"].mean())
        with open(
            f"results/afc_validation/{str.lower(r)}_death_age_mean.json", "w"
        ) as json_file:
            json.dump(
                total_trace[total_trace["race"] == r]["years_to_death"].mean(),
                json_file,
            )
        with open(
            f"results/afc_validation/{str.lower(r)}_death_age_se.json", "w"
        ) as json_file:
            json.dump(
                total_trace[total_trace["race"] == r]["years_to_death"].std()
                / len(total_trace[total_trace["race"] == r]),
                json_file,
            )

    with open("results/afc_validation/sick_percent.json", "w") as json_file:
        json.dump(total_trace["was_sick"].mean(), json_file)
    with open("results/afc_validation/sick_percent_se.json", "w") as json_file:
        json.dump(total_trace["was_sick"].std() / len(total_trace), json_file)
    # number of years sick
    with open("results/afc_validation/years_sick_mean.json", "w") as json_file:
        json.dump(
            total_trace[total_trace["was_sick"] == 1]["years_sick"].mean(), json_file
        )
    with open("results/afc_validation/years_sick_se.json", "w") as json_file:
        json.dump(
            total_trace[total_trace["was_sick"] == 1]["years_sick"].std()
            / len(total_trace[total_trace["was_sick"] == 1]),
            json_file,
        )


def export_validation_SDI_histogram():
    client = bq_client()

    # Reads from the SBP_DBP_values_ordered_nonmeds_followups built table
    # (see code/SQL/hypertension_sql_queries.sql); the self-join on max(row_number)
    # and the SDI/month_difference/age/max_number filters are baked into that table.
    query = f"""
    SELECT * from {table("SBP_DBP_values_ordered_nonmeds_followups")}
    ORDER BY person_id
    LIMIT 1000000
    """

    query_job = client.query(query)
    df = query_job.to_dataframe()

    df.columns = [
        "person_id",
        "female",
        "age",
        "race",
        "ethnicity",
        "SDI",
        "last_systolic_bp",
        "last_diastolic_bp",
        "last_date",
        "this_date",
        "systolic BP",
        "diastolic BP",
        "row_number",
        "days_difference",
        "month_difference",
        "first_exposure",
        "last_exposure",
        "days_to_medicine",
        "person_id_1",
        "max_number",
    ]

    mask = (df["race"] == "Black or African American") & (
        df["ethnicity"] == "Not Hispanic or Latino"
    )
    df["black"] = mask.astype(int)

    mask = (df["race"] == "White") & (df["ethnicity"] == "Not Hispanic or Latino")
    df["white"] = mask.astype(int)

    starting_age = 40
    cohort_size = 10000
    cohort_size_half = int(cohort_size / 2)

    person_df = df[df["age"] == starting_age][
        ["person_id", "age", "black", "white", "female", "SDI"]
    ].drop_duplicates()
    person_df = person_df.reset_index()
    person_df = person_df.drop(columns=["index"])

    person_df_black = person_df[person_df["black"] == 1]
    person_df_black_sample = person_df_black.sample(
        n=cohort_size_half, replace=True
    ).reset_index()

    person_df_white = person_df[person_df["white"] == 1]
    person_df_white_sample = person_df_white.sample(
        n=cohort_size_half, replace=True
    ).reset_index()

    sampled_df = pd.concat([person_df_black_sample, person_df_white_sample])

    with open("data_and_models/afc_outputs/social/afc_sdi_mean.json", "w") as json_file:
        json.dump(sampled_df["SDI"].mean(), json_file)
    with open("data_and_models/afc_outputs/social/afc_sdi_std.json", "w") as json_file:
        json.dump(sampled_df["SDI"].std(), json_file)

    plt.figure(figsize=(5, 4))
    plt.hist(sampled_df["SDI"])
    plt.xlabel("SDI")
    plt.ylabel("Frequency")
    plt.savefig(
        "figures/supplement/SDI_histogram_afc.png", dpi=300, bbox_inches="tight"
    )


def create_combined_kaplan_meier_curve():
    client = bq_client()

    age_list = [40, 46, 52, 58, 64, 70]
    data_frames = []
    for a in age_list:
        query = f"select * from `{table(f'patients_{a}y_eligible')}`"
        query_job = client.query(query)
        df = query_job.to_dataframe()
        df["entry_age"] = a
        df["simulated_death_age"] = df.apply(
            lambda row: simulate_death_age(
                "NHB" if row["NHB"] == 1 else "NHW",
                row["gender"],
                int(row["entry_age"]),
            ),
            axis=1,
        )
        # Event occurs before death or end of follow-up
        df["event"] = df.apply(
            lambda row: (
                pd.notna(row["age"])
                and row["entry_age"]
                < row["age"]
                <= min(row["follow_up_age"], row["simulated_death_age"])
            ),
            axis=1,
        ).astype(int)

        # Exit is minimum of age at treatment or last follow-up
        df["exit_age"] = df.apply(
            lambda row: (
                row["age"]
                if row["event"] == 1
                else min(row["follow_up_age"], row["simulated_death_age"])
            ),
            axis=1,
        )

        N_nhb = (df["NHB"] == 1).sum()
        total = len(df)
        N_nhw = total - N_nhb

        # Assign weights
        df["weight"] = df["NHB"].apply(
            lambda g: (0.5 / N_nhb) if g == 1 else (0.5 / N_nhw)
        )

        data_frames.append(df)
    combined_df = pd.concat(data_frames)
    combined_df = combined_df.sort_values("entry_age")
    combined_df = combined_df.drop_duplicates(subset="person_id", keep="first")

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=combined_df["exit_age"],
        event_observed=combined_df["event"],
        entry=combined_df["entry_age"],
        weights=combined_df["weight"],
        label="Combined survival",
    )

    kmf.cumulative_density_.to_csv(
        "data_and_models/afc_outputs/standard/treatment_cumulative_density.csv",
        index=False,
    )


def create_combined_kaplan_meier_curve_multiple(n_simulations=100):
    client = bq_client()

    age_list = [40, 46, 52, 58, 64, 70]
    data_frames = []
    for a in age_list:
        query = f"select * from `{table(f'patients_{a}y_eligible')}`"
        df = client.query(query).to_dataframe()
        df["entry_age"] = a
        data_frames.append(df)

    raw_df = pd.concat(data_frames)
    raw_df = raw_df.sort_values("entry_age")
    raw_df = raw_df.drop_duplicates(subset="person_id", keep="first")

    N_nhb = (raw_df["NHB"] == 1).sum()
    N_nhw = len(raw_df) - N_nhb
    raw_df["weight"] = raw_df["NHB"].apply(
        lambda g: (0.5 / N_nhb) if g == 1 else (0.5 / N_nhw)
    )

    all_curves = []

    for sim in range(n_simulations):
        df = raw_df.copy()

        df["simulated_death_age"] = df.apply(
            lambda row: simulate_death_age(
                "NHB" if row["NHB"] == 1 else "NHW",
                row["gender"],
                int(row["entry_age"]),
            ),
            axis=1,
        )

        df["event"] = df.apply(
            lambda row: (
                pd.notna(row["age"])
                and row["entry_age"]
                < row["age"]
                <= min(row["follow_up_age"], row["simulated_death_age"])
            ),
            axis=1,
        ).astype(int)

        df["exit_age"] = df.apply(
            lambda row: (
                row["age"]
                if row["event"] == 1
                else min(row["follow_up_age"], row["simulated_death_age"])
            ),
            axis=1,
        )

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=df["exit_age"],
            event_observed=df["event"],
            entry=df["entry_age"],
            weights=df["weight"],
            label="Combined survival",
        )

        all_curves.append(kmf.cumulative_density_["Combined survival"])

    ##sometimes the steps may differ by age, so we need to do an interpolation
    time_grid = np.linspace(
        raw_df["entry_age"].min(),
        raw_df["exit_age"].max()
        if "exit_age" in raw_df.columns
        else raw_df["follow_up_age"].max(),
        500,
    )

    interpolated = []
    for curve in all_curves:
        reindexed = (
            curve.reindex(curve.index.union(time_grid)).ffill().reindex(time_grid)
        )
        interpolated.append(reindexed.values)

    sim_matrix = np.vstack(interpolated)  # shape: (n_simulations, 500)

    summary = pd.DataFrame(
        {
            "time": time_grid,
            "median": np.nanpercentile(sim_matrix, 50, axis=0),
            "ci_lower": np.nanpercentile(sim_matrix, 2.5, axis=0),
            "ci_upper": np.nanpercentile(sim_matrix, 97.5, axis=0),
        }
    )

    summary.to_csv(
        "data_and_models/afc_outputs/standard/treatment_cumulative_density_multiple.csv",
        index=False,
    )
    return summary


def main():
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)

    parser = ArgumentParser()
    parser.add_argument("-f", dest="folder", required=True, help="validation folder")
    args = parser.parse_args()
    validation_folder = args.folder

    # Ensure all output directories exist before any write.
    os.makedirs("results/afc_validation", exist_ok=True)
    os.makedirs(f"{overall_folder}/figures/supplement", exist_ok=True)
    os.makedirs(f"{overall_folder}/data_and_models/afc_outputs/standard", exist_ok=True)
    os.makedirs(f"{overall_folder}/data_and_models/afc_outputs/social", exist_ok=True)

    total_trace, sampled_cohort = read_in_afc_validation_files(
        overall_folder, validation_folder
    )

    create_afc_validation_summary_stats(sampled_cohort)
    CYCLE_LENGTH = 1 / 12.0
    cycles = int(60 / CYCLE_LENGTH) + 5
    N_arr, E_arr, S1_arr, S2_arr = create_trace_arrs(total_trace, cycles)
    create_validation_outputs(total_trace, cycles, N_arr, E_arr, S1_arr, S2_arr)
    export_validation_death_sick_statistics(total_trace)

    folder = "final_cohort"
    sampled_cohort = pd.read_csv(
        f"{overall_folder}/results/{folder}/cohort_files/standard/sampled_cohort.csv"
    )
    conduct_KS_test_measurement_intervals(overall_folder, folder, sampled_cohort)
    export_validation_SDI_histogram()
    create_combined_kaplan_meier_curve()


if __name__ == "__main__":
    main()

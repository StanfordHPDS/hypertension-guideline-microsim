import os
import pandas as pd
import numpy as np
from bq_config import bq_client, table
from scipy.stats import norm
from argparse import ArgumentParser
from cohort_sampling_NHANES import (
    return_systolic_diastolic_quantiles,
)
from numeric import round_to_nearest_0_01


def simulate_person_cohort(cohort_size=10000, seed=42):
    rng = np.random.default_rng(seed)

    female = rng.binomial(1, 0.52, size=cohort_size)
    age = np.full(cohort_size, 40, dtype=float)

    race = rng.choice(
        ["White", "Black or African American", "Asian", "Other"],
        size=cohort_size,
        p=[0.50, 0.20, 0.15, 0.15],
    )

    ethnicity = rng.choice(
        ["Not Hispanic or Latino", "Hispanic or Latino"],
        size=cohort_size,
        p=[0.80, 0.20],
    )

    SDI = rng.uniform(0.05, 0.95, size=cohort_size)

    systolic = 118 + 6 * female + 10 * SDI + rng.normal(0, 12, size=cohort_size)
    diastolic = 74 + 3 * female + 5 * SDI + rng.normal(0, 8, size=cohort_size)

    systolic = np.clip(systolic, 85, 240)
    diastolic = np.clip(diastolic, 45, 140)

    df = pd.DataFrame(
        {
            "age": age,
            "female": female,
            "SDI": SDI,
            "race": race,
            "ethnicity": ethnicity,
            "systolic BP": systolic,
            "diastolic BP": diastolic,
        }
    )

    BP_stage_conditions = [
        (df["systolic BP"] >= 140) | (df["diastolic BP"] >= 90),
        (df["systolic BP"] >= 130) | (df["diastolic BP"] >= 80),
        (df["systolic BP"] >= 120) & (df["diastolic BP"] < 80),
    ]
    BP_stage_values = ["S2", "S1", "E"]
    df["BP_stage"] = np.select(BP_stage_conditions, BP_stage_values, default="N")

    df["black"] = (
        (df["race"] == "Black or African American")
        & (df["ethnicity"] == "Not Hispanic or Latino")
    ).astype(int)

    df["white"] = (
        (df["race"] == "White") & (df["ethnicity"] == "Not Hispanic or Latino")
    ).astype(int)

    df["hispanic"] = (df["ethnicity"] == "Hispanic or Latino").astype(int)
    df["age_rounded"] = df["age"].round(3).astype(float)

    return df


def read_in_afc_data():
    client = bq_client()

    # Reads from the SBP_DBP_values_nonmeds_age40_with_row_counts built table
    # (see code/SQL/hypertension_sql_queries.sql); the age=40, non-null SDI, and
    # per-person max(row_num) < 1000 cap are baked into that table.
    query = f"""
    SELECT * from {table("SBP_DBP_values_nonmeds_age40_with_row_counts")}
    """

    query_job = client.query(query)
    df = query_job.to_dataframe()

    df.columns = [
        "person_id",
        "measurement_datetime",
        "measurement_concept_id",
        "systolic BP",
        "diastolic BP",
        "row_num",
        "female",
        "age",
        "race",
        "ethnicity",
        "SDI",
        "person_id_1",
        "max_number",
    ]

    # make some modifications
    BP_stage_conditions = [
        (df["systolic BP"] >= 140) | (df["diastolic BP"] >= 90),
        (df["systolic BP"] >= 130) | (df["diastolic BP"] >= 80),
        (df["systolic BP"] >= 120) & (df["diastolic BP"] < 80),
    ]
    BP_stage_values = ["S2", "S1", "E"]
    df["BP_stage"] = np.select(BP_stage_conditions, BP_stage_values, default="N")

    df["race"] = df["race"].replace({"Black": "Black or African American"})
    df["race"] = df["race"].replace({"Vietnamese": "Asian"})
    df["race"] = df["race"].replace({"African American": "Black or African American"})
    df["race"] = df["race"].replace(
        {"Other Pacific Islander": "Native Hawaiian or Other Pacific Islander"}
    )

    mask = (df["race"] == "Black or African American") & (
        df["ethnicity"] == "Not Hispanic or Latino"
    )
    df["black"] = mask.astype(int)

    mask = (df["race"] == "White") & (df["ethnicity"] == "Not Hispanic or Latino")
    df["white"] = mask.astype(int)

    mask = df["ethnicity"] == "Hispanic or Latino"
    df["hispanic"] = mask.astype(int)

    return df


def create_person_df(df, cohort_size, rng):
    person_df = df[
        [
            "person_id",
            "age",
            "black",
            "white",
            "hispanic",
            "female",
            "SDI",
            "systolic BP",
            "diastolic BP",
            "BP_stage",
        ]
    ].drop_duplicates()
    person_df = person_df.reset_index()
    person_df = person_df.drop(columns=["index"])

    sampled_df = (
        person_df[
            [
                "age",
                "black",
                "white",
                "hispanic",
                "female",
                "SDI",
                "systolic BP",
                "diastolic BP",
                "BP_stage",
            ]
        ]
        .sample(n=cohort_size, replace=True, random_state=rng)
        .reset_index()
    )
    sampled_df["age_rounded"] = sampled_df["age"].round(3).astype(float)
    return sampled_df


def export_validation_systolic_diastolic_changing_quantiles(
    social, cohort_size, cohort_size_half, overall_folder, folder, rng
):
    correlations = []
    for i in range(11):
        if social:
            correlations.append(
                pd.read_csv(
                    f"{overall_folder}/data_and_models/afc_outputs/social/AFC_age_group_correlations/saved_{str(i)}.csv"
                ).values
            )
        else:
            correlations.append(
                pd.read_csv(
                    f"{overall_folder}/data_and_models/afc_outputs/standard/AFC_age_group_correlations/saved_{str(i)}.csv"
                ).values
            )

    ##for age group buckets in my data, I need to append the 4-year correlations together to create correlated quantiles that determine an individual's blood pressure trajectory over the lifetime horizon. The individual's quantile can change ever year
    for a in range(16):
        # for ages older than 80, we are going to use the same correlation matrix
        if a >= 10:
            this_correlation = correlations[8]
        else:
            this_correlation = correlations[a]

        mean = np.zeros(this_correlation.shape[0])  # Zero mean
        cov = (
            this_correlation  # Covariance matrix is the same as the correlation matrix
        )
        num_samples = cohort_size  # Number of samples
        # sample correlated quantiles between 0 and 1 for all individuals during this 4-year window
        rv = rng.multivariate_normal(mean, cov, size=num_samples)
        transformed_vars = []
        for n in range(num_samples):
            transformed_vars.append(norm.cdf(rv[n]))
        transformed_vars = np.array(transformed_vars)
        transformed_vars = np.vectorize(round_to_nearest_0_01)(transformed_vars)
        # sort them according to the final year of the 4-year window
        sorted_arr = transformed_vars[transformed_vars[:, 0].argsort()]

        systolic_quantiles = sorted_arr[:, 0:4]
        diastolic_quantiles = sorted_arr[:, 4:]

        if a == 0:
            total_arr = sorted_arr
            total_sys_arr = systolic_quantiles
            total_dia_arr = diastolic_quantiles

        else:
            # append them together so we get lifetime quantiles
            total_arr = np.concatenate((total_arr, sorted_arr), axis=1)
            total_sys_arr = np.concatenate((total_sys_arr, systolic_quantiles), axis=1)
            total_dia_arr = np.concatenate((total_dia_arr, diastolic_quantiles), axis=1)

    sampled_quantiles = pd.concat(
        [
            pd.DataFrame(total_sys_arr),
            pd.DataFrame(
                total_dia_arr,
                columns=range(len(total_sys_arr[0]), len(total_sys_arr[0]) * 2),
            ),
        ],
        axis=1,
    )
    sampled_quantiles = sampled_quantiles.clip(lower=0.01, upper=0.99)
    sampled_quantiles = sampled_quantiles.sort_values(by=0)
    sampled_quantiles = sampled_quantiles.reset_index()
    sampled_quantiles = sampled_quantiles.drop(columns=["index"])

    sampled_quantiles[sampled_quantiles.columns[: len(total_sys_arr[0])]].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_systolic_quantiles.csv",
        index=False,
    )
    sampled_quantiles[sampled_quantiles.columns[len(total_sys_arr[0]) :]].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_diastolic_quantiles.csv",
        index=False,
    )

    sampled_quantiles[sampled_quantiles.columns[: len(total_sys_arr[0])]].iloc[
        :cohort_size_half
    ].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_systolic_quantiles_1.csv",
        index=False,
    )

    sampled_quantiles[sampled_quantiles.columns[: len(total_sys_arr[0])]].iloc[
        cohort_size_half:
    ].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_systolic_quantiles_2.csv",
        index=False,
    )

    sampled_quantiles[sampled_quantiles.columns[len(total_sys_arr[0]) :]].iloc[
        :cohort_size_half
    ].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_diastolic_quantiles_1.csv",
        index=False,
    )

    sampled_quantiles[sampled_quantiles.columns[len(total_sys_arr[0]) :]].iloc[
        cohort_size_half:
    ].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_diastolic_quantiles_2.csv",
        index=False,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", dest="folder", required=True, help="cohort path")
    parser.add_argument("-n", dest="cohort_size", required=True, help="cohort_size")
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Master seed for the validation cohort RNG (default: 42).",
    )

    args = parser.parse_args()
    folder = args.folder
    starting_age = 40
    cohort_size = int(args.cohort_size)
    cohort_size_half = int(cohort_size / 2)
    social = False
    rng = np.random.default_rng(args.seed)

    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)

    df = read_in_afc_data()
    sampled_df = create_person_df(df, cohort_size, rng)

    ##MARIKA'S TEST DATA
    # sampled_df = simulate_person_cohort(cohort_size)

    sampled_df = return_systolic_diastolic_quantiles(
        overall_folder, sampled_df, starting_age, social
    )

    ## TO CONFIRM: DO I NOT DO THE NORMAL HERE?
    # sampled_df = assign_normal_quantiles(
    #    overall_folder, sampled_df, starting_age, social
    # )

    sampled_df = sampled_df.reset_index()

    # 2**32 is the widest seed value np.random.seed accepts (it raises
    # ValueError above that); 2**32 keeps the expected per-person seed
    # collision count on a 100k cohort near 1, vs. ~5000 at the previous 1M.
    random_seeds = rng.integers(0, 2**32, size=cohort_size, dtype=np.uint32)
    sampled_df["random_seed"] = pd.Series(random_seeds)

    # sort them in order of 0 to 1
    sampled_df = sampled_df.sort_values(by="age_systolic_percentile")
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df = sampled_df.drop(columns=["index"])

    os.makedirs(f"{overall_folder}/validation_results/{folder}/", exist_ok=True)

    sampled_df.to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_cohort.csv", index=False
    )
    sampled_df.iloc[:cohort_size_half].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_cohort_1.csv",
        index=False,
    )
    sampled_df.iloc[cohort_size_half:].to_csv(
        f"{overall_folder}/validation_results/{folder}/sampled_cohort_2.csv",
        index=False,
    )

    export_validation_systolic_diastolic_changing_quantiles(
        social, cohort_size, cohort_size_half, overall_folder, folder, rng
    )


if __name__ == "__main__":
    main()

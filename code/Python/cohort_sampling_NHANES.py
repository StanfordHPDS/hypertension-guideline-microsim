import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from argparse import ArgumentParser
from bp_generation import (
    MEASUREMENT_FREQ_COLS_SOCIAL,
    MEASUREMENT_FREQ_COLS_STANDARD,
    generate_hypertension_stage,
    generate_measurement_frequency,
    generate_measurement_frequency_SDI,
    generate_new_DBP_value,
    generate_new_DBP_value_SDI,
    generate_new_SBP_value,
    generate_new_SBP_value_SDI,
    load_measurement_models,
    load_quantile_models,
    load_splines,
)
from demographics import (
    create_cohort_demographics,
    create_cohort_demographics_by_race,
)
from model_fitting import build_predictor_df, parallel_predict_models, raise_if_nan
from numeric import round_to_nearest_0_01
from paths import overall_folder


def read_in_NHANES_data(NHANES_file):
    # this is simulated within R using the NHANES multivariate copula code
    NHANES_cohort = pd.read_csv(
        f"{overall_folder}/data_and_models/nhanes_data_cohorts/{NHANES_file}_sdi.csv"
    )
    NHANES_cohort.columns = [
        "age",
        "female",
        "black",
        "place",
        "insurance",
        "poverty_ratio",
        "systolic BP",
        "diastolic BP",
        "Poverty_LT100",
        "Education_LT12years",
        "HH_Renter_Occupied",
        "NonEmployed",
        "HH_Crowding",
        "Single_Parent_Fam",
        "visits",
        "SDI",
    ]

    cohort_size = len(NHANES_cohort)
    # 2**32 is the widest seed value np.random.seed accepts (it raises
    # ValueError above that); 2**32 keeps the expected per-person seed
    # collision count on a 100k cohort near 1, vs. ~5000 at the previous 1M.
    random_seeds = np.random.randint(0, 2**32, size=cohort_size, dtype=np.uint32)
    NHANES_cohort["random_seed"] = pd.Series(random_seeds)
    NHANES_cohort["age_rounded"] = [40.000 for i in range(len(NHANES_cohort))]

    return NHANES_cohort


def assign_systolic_quantile(
    df, systolic_splines, sb_quantile_models, quantiles, social
):
    df_sbp = df.merge(
        systolic_splines,
        left_on=["age_rounded", "female"],
        right_on=["age", "female"],
        how="left",
        suffixes=("", "_spline"),
    )
    # Construct a predictors DataFrame for SBP.
    spline_cols_sbp = list(systolic_splines.columns[2:])
    X_sbp = build_predictor_df(df_sbp, spline_cols_sbp, social)
    for col in spline_cols_sbp:
        X_sbp[col] = df_sbp[col]

    pred_matrix_sbp = parallel_predict_models(sb_quantile_models, X_sbp)

    raise_if_nan(df_sbp["systolic BP"].values, "systolic BP")
    diff_matrix_sbp = np.abs(
        df_sbp["systolic BP"].values.reshape(-1, 1) - pred_matrix_sbp
    )
    min_index_sbp = np.argmin(diff_matrix_sbp, axis=1)
    df["age_systolic_percentile"] = np.array(quantiles)[min_index_sbp]
    return df


def assign_diastolic_quantile(
    df, diastolic_splines, db_quantile_models, quantiles, social
):
    df_dbp = df.merge(
        diastolic_splines,
        left_on=["age_rounded", "female"],
        right_on=["age", "female"],
        how="left",
        suffixes=("", "_spline"),
    )

    spline_cols_dbp = list(diastolic_splines.columns[2:])
    X_dbp = build_predictor_df(df_dbp, spline_cols_dbp, social)

    for col in spline_cols_dbp:
        X_dbp[col] = df_dbp[col]

    pred_matrix_dbp = parallel_predict_models(db_quantile_models, X_dbp)

    raise_if_nan(df_dbp["diastolic BP"].values, "diastolic BP")
    diff_matrix_dbp = np.abs(
        df_dbp["diastolic BP"].values.reshape(-1, 1) - pred_matrix_dbp
    )
    min_index_dbp = np.argmin(diff_matrix_dbp, axis=1)
    df["age_diastolic_percentile"] = np.array(quantiles)[min_index_dbp]
    return df


def return_systolic_diastolic_quantiles(overall_folder, df, starting_age, social):
    quantiles = np.arange(0.01, 1.0, 0.01)
    quantiles = [round(x, 2) for x in quantiles]
    cohort_size = len(df)

    sb_quantile_models, db_quantile_models = load_quantile_models(
        overall_folder, social, quantiles
    )
    systolic_splines, diastolic_splines = load_splines(overall_folder, social)

    df = assign_systolic_quantile(
        df, systolic_splines, sb_quantile_models, quantiles, social
    )
    df = assign_diastolic_quantile(
        df, diastolic_splines, db_quantile_models, quantiles, social
    )

    if social:
        df["predicted_systolic_bp"] = np.array(
            [
                generate_new_SBP_value_SDI(
                    sb_quantile_models,
                    systolic_splines,
                    quantiles.index(df["age_systolic_percentile"].iloc[i]),
                    starting_age,
                    df["female"].iloc[i],
                    df["SDI"].iloc[i],
                )
                for i in range(cohort_size)
            ]
        )
        df["predicted_diastolic_bp"] = np.array(
            [
                generate_new_DBP_value_SDI(
                    db_quantile_models,
                    diastolic_splines,
                    quantiles.index(df["age_diastolic_percentile"].iloc[i]),
                    starting_age,
                    df["female"].iloc[i],
                    df["SDI"].iloc[i],
                )
                for i in range(cohort_size)
            ]
        )
    else:
        df["predicted_systolic_bp"] = np.array(
            [
                generate_new_SBP_value(
                    sb_quantile_models,
                    systolic_splines,
                    quantiles.index(df["age_systolic_percentile"].iloc[i]),
                    starting_age,
                    df["female"].iloc[i],
                )
                for i in range(cohort_size)
            ]
        )
        df["predicted_diastolic_bp"] = np.array(
            [
                generate_new_DBP_value(
                    db_quantile_models,
                    diastolic_splines,
                    quantiles.index(df["age_diastolic_percentile"].iloc[i]),
                    starting_age,
                    df["female"].iloc[i],
                )
                for i in range(cohort_size)
            ]
        )
    # Compute the BP stage
    df["predicted_BP_stage"] = np.array(
        [
            generate_hypertension_stage(
                df["predicted_systolic_bp"].iloc[i],
                df["predicted_diastolic_bp"].iloc[i],
            )
            for i in range(cohort_size)
        ]
    )
    df = df.reset_index(drop=True)

    return df


def assign_normal_quantiles(overall_folder, df, starting_age, social):
    cohort_size = len(df)

    mean = np.zeros(2)
    cov = [[1, 0.66], [0.66, 1]]
    rv = np.random.multivariate_normal(mean, cov, size=cohort_size)
    these_first_quantiles = norm.cdf(rv)
    these_first_quantiles = [
        [
            min(max(round_to_nearest_0_01(these_first_quantiles[i][0]), 0.01), 0.99),
            min(max(round_to_nearest_0_01(these_first_quantiles[i][1]), 0.01), 0.99),
        ]
        for i in range(cohort_size)
    ]

    normal_quantiles = pd.DataFrame(
        these_first_quantiles,
        columns=["normal_systolic_percentile", "normal_diastolic_percentile"],
    )
    normal_quantiles = normal_quantiles.sort_values(by="normal_systolic_percentile")
    normal_quantiles = normal_quantiles.reset_index()
    normal_quantiles = normal_quantiles.drop(columns=["index"])

    df = df.sort_values(by="age_systolic_percentile")
    df = df.reset_index()
    df = df.drop(columns=["index"])

    df = pd.concat([df, normal_quantiles], axis=1)
    quantiles = np.arange(0.01, 1.0, 0.01)
    quantiles = [round(x, 2) for x in quantiles]

    sb_quantile_models, db_quantile_models = load_quantile_models(
        overall_folder, social, quantiles
    )
    systolic_splines, diastolic_splines = load_splines(overall_folder, social)

    if social:
        df["predicted_systolic_bp"] = [
            generate_new_SBP_value_SDI(
                sb_quantile_models,
                systolic_splines,
                quantiles.index(df["normal_systolic_percentile"].iloc[i]),
                starting_age,
                df["female"].iloc[i],
                df["SDI"].iloc[i],
            )
            for i in range(cohort_size)
        ]

        df["predicted_diastolic_bp"] = [
            generate_new_DBP_value_SDI(
                db_quantile_models,
                diastolic_splines,
                quantiles.index(df["normal_diastolic_percentile"].iloc[i]),
                starting_age,
                df["female"].iloc[i],
                df["SDI"].iloc[i],
            )
            for i in range(cohort_size)
        ]
    else:
        df["predicted_systolic_bp"] = [
            generate_new_SBP_value(
                sb_quantile_models,
                systolic_splines,
                quantiles.index(df["normal_systolic_percentile"].iloc[i]),
                starting_age,
                df["female"].iloc[i],
            )
            for i in range(cohort_size)
        ]

        df["predicted_diastolic_bp"] = [
            generate_new_DBP_value(
                db_quantile_models,
                diastolic_splines,
                quantiles.index(df["normal_diastolic_percentile"].iloc[i]),
                starting_age,
                df["female"].iloc[i],
            )
            for i in range(cohort_size)
        ]

    df["predicted_BP_stage"] = [
        generate_hypertension_stage(
            df["predicted_systolic_bp"].iloc[i],
            df["predicted_diastolic_bp"].iloc[i],
        )
        for i in range(cohort_size)
    ]

    df["age_systolic_percentile"] = df["normal_systolic_percentile"]
    df["age_diastolic_percentile"] = df["normal_diastolic_percentile"]

    return df


def assign_measurement_frequency(overall_folder, NHANES_cohort, social):
    quantiles = np.arange(0.01, 1.0, 0.01)
    quantiles = [round(x, 2) for x in quantiles]

    measurement_frequency_models = load_measurement_models(
        overall_folder, social, quantiles
    )
    measure_X = pd.DataFrame(
        {
            "intercept": 1,
            "female": NHANES_cohort["female"],
            "age": NHANES_cohort["age"],
            "last_systolic_bp": NHANES_cohort["systolic BP"],
            "last_diastolic_bp": NHANES_cohort["diastolic BP"],
        }
    )
    if social:
        measure_X["SDI"] = NHANES_cohort["SDI"]
    cols = MEASUREMENT_FREQ_COLS_SOCIAL if social else MEASUREMENT_FREQ_COLS_STANDARD
    measure_X = measure_X[cols]

    pred_matrix = parallel_predict_models(measurement_frequency_models, measure_X)

    # sample number of months since last visit
    sample_no_months = [0 for i in range(len(NHANES_cohort))]
    for i in range(len(NHANES_cohort)):
        if NHANES_cohort["visits"].iloc[i] != 0:
            sampled_months = np.random.randint(1, 13, NHANES_cohort["visits"].iloc[i])
            sample_no_months[i] = np.min(sampled_months)
        else:
            sample_no_months[i] = np.random.randint(12, 24, 1)[0]

    NHANES_cohort["month_difference"] = pd.Series(sample_no_months)
    # find the percentile that matches most closely to their time to the next visit
    month_diff_array = NHANES_cohort["month_difference"].to_numpy()
    raise_if_nan(month_diff_array, "month_difference")
    diff_matrix = np.abs(pred_matrix - month_diff_array[:, np.newaxis])
    min_index = np.argmin(diff_matrix, axis=1)
    NHANES_cohort["measurement_freq_percentile"] = np.array(quantiles)[min_index]

    pred_interval = [0 for i in range(len(NHANES_cohort))]
    for i in range(len(NHANES_cohort)):
        if social:
            pred_interval[i] = round(
                generate_measurement_frequency_SDI(
                    measurement_frequency_models,
                    quantiles.index(
                        NHANES_cohort["measurement_freq_percentile"].iloc[i]
                    ),
                    NHANES_cohort["age"].iloc[i],
                    NHANES_cohort["female"].iloc[i],
                    NHANES_cohort["SDI"].iloc[i],
                    NHANES_cohort["systolic BP"].iloc[i],
                    NHANES_cohort["diastolic BP"].iloc[i],
                )
            )
        else:
            pred_interval[i] = round(
                generate_measurement_frequency(
                    measurement_frequency_models,
                    quantiles.index(
                        NHANES_cohort["measurement_freq_percentile"].iloc[i]
                    ),
                    NHANES_cohort["age"].iloc[i],
                    NHANES_cohort["female"].iloc[i],
                    NHANES_cohort["systolic BP"].iloc[i],
                    NHANES_cohort["diastolic BP"].iloc[i],
                )
            )

    NHANES_cohort["predicted_month_difference"] = pd.Series(pred_interval)
    return NHANES_cohort


def return_systolic_diastolic_changing_quantiles(
    social, cohort_size, overall_folder, folder
):
    # estimated correlations from the data
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

    ##for age group buckets in my data, I need to append the 4-year correlations together to create correlations for the lifetime horizon
    # I did not update this code to be vectorized yet, perhaps there can be a solution to this as well
    for a in range(16):
        if a >= 10:
            this_correlation = correlations[8]
        else:
            this_correlation = correlations[a]

        mean = np.zeros(this_correlation.shape[0])  # Zero mean
        cov = (
            this_correlation  # Covariance matrix is the same as the correlation matrix
        )
        num_samples = cohort_size  # Number of samples
        rv = np.random.multivariate_normal(mean, cov, size=num_samples)
        transformed_vars = []
        for n in range(num_samples):
            transformed_vars.append(norm.cdf(rv[n]))
        transformed_vars = np.array(transformed_vars)
        transformed_vars = np.vectorize(round_to_nearest_0_01)(transformed_vars)
        sorted_arr = transformed_vars[transformed_vars[:, 0].argsort()]

        systolic_quantiles = sorted_arr[:, 0:4]
        diastolic_quantiles = sorted_arr[:, 4:]

        if a == 0:
            total_arr = sorted_arr
            total_sys_arr = systolic_quantiles
            total_dia_arr = diastolic_quantiles

        else:
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
    if social:
        sampled_quantiles[sampled_quantiles.columns[: len(total_sys_arr[0])]].to_csv(
            f"{overall_folder}/results/{folder}/cohort_files/social/sampled_systolic_quantiles.csv",
            index=False,
        )
        sampled_quantiles[sampled_quantiles.columns[len(total_sys_arr[0]) :]].to_csv(
            f"{overall_folder}/results/{folder}/cohort_files/social/sampled_diastolic_quantiles.csv",
            index=False,
        )
    else:
        sampled_quantiles[sampled_quantiles.columns[: len(total_sys_arr[0])]].to_csv(
            f"{overall_folder}/results/{folder}/cohort_files/standard/sampled_systolic_quantiles.csv",
            index=False,
        )
        sampled_quantiles[sampled_quantiles.columns[len(total_sys_arr[0]) :]].to_csv(
            f"{overall_folder}/results/{folder}/cohort_files/standard/sampled_diastolic_quantiles.csv",
            index=False,
        )


def main():
    np.random.seed(42)
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)

    parser = ArgumentParser()
    parser.add_argument("-f", dest="folder", required=True, help="cohort path")
    parser.add_argument(
        "-d", dest="NHANES_file", required=True, help="NHANES file name"
    )
    parser.add_argument(
        "-sff", dest="sff", required=True, help='social factors framework ("1"or "0")'
    )
    args = parser.parse_args()
    folder = args.folder
    NHANES_file = args.NHANES_file
    sff = args.sff
    social = sff == "1"

    starting_age = 40
    NHANES_cohort = read_in_NHANES_data(NHANES_file)
    NHANES_cohort = return_systolic_diastolic_quantiles(
        overall_folder, NHANES_cohort, starting_age, social
    )
    NHANES_cohort = assign_normal_quantiles(
        overall_folder, NHANES_cohort, starting_age, social
    )
    NHANES_cohort = assign_measurement_frequency(overall_folder, NHANES_cohort, social)

    os.makedirs(f"{overall_folder}/results/{folder}/cohort_files/", exist_ok=True)

    if social:
        os.makedirs(
            f"{overall_folder}/results/{folder}/cohort_files/social/", exist_ok=True
        )
        NHANES_cohort.to_csv(
            f"{overall_folder}/results/{folder}/cohort_files/social/sampled_cohort.csv",
            index=False,
        )
    else:
        os.makedirs(
            f"{overall_folder}/results/{folder}/cohort_files/standard/", exist_ok=True
        )
        NHANES_cohort.to_csv(
            f"{overall_folder}/results/{folder}/cohort_files/standard/sampled_cohort.csv",
            index=False,
        )

    return_systolic_diastolic_changing_quantiles(
        social, len(NHANES_cohort), overall_folder, folder
    )

    ##output demographic information for this cohort
    create_cohort_demographics(NHANES_cohort, folder, social)
    demo_dict_NHB, demo_dict_NHW = create_cohort_demographics_by_race(
        NHANES_cohort, folder, social
    )

    if social:
        # histogram of the NHANES cohort SDI distribution
        plt.figure(figsize=(5, 4))
        plt.hist(NHANES_cohort["SDI"])
        plt.xlabel("SDI")
        plt.ylabel("Frequency")

        # Save the histogram to a file
        os.makedirs(f"{overall_folder}/figures/supplement", exist_ok=True)
        plt.savefig(
            f"{overall_folder}/figures/supplement/SDI_histogram_cohort.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()

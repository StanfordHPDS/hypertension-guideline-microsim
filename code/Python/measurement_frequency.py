import os
import pandas as pd
import numpy as np
from bq_config import bq_client, table
import statsmodels.api as sm
from bp_generation import MEASUREMENT_FREQ_COLS_SOCIAL, MEASUREMENT_FREQ_COLS_STANDARD
from model_fitting import parallel_predict_models, quant_goodfit, raise_if_nan
from paths import overall_folder
from concurrent.futures import as_completed, ThreadPoolExecutor
import time
import json
from argparse import ArgumentParser


def read_in_afc_data():
    client = bq_client()

    query = f"""
    SELECT * from {table("SBP_DBP_values_ordered_nonmeds_sample_1m")}
    """

    query_job = client.query(query)
    df = query_job.to_dataframe()

    df["intercept"] = pd.Series([1 for i in range(len(df))])
    return df


def export_measurement_mean_sd(df, social):
    month_difference_mean = df["month_difference"].mean()
    month_difference_std = df["month_difference"].std()

    if social:
        with open(
            f"{overall_folder}/data_and_models/afc_outputs/social/month_difference_mean.json",
            "w",
        ) as json_file:
            json.dump(month_difference_mean, json_file)
        with open(
            f"{overall_folder}/data_and_models/afc_outputs/social/month_difference_std.json",
            "w",
        ) as json_file:
            json.dump(month_difference_std, json_file)
    else:
        with open(
            f"{overall_folder}/data_and_models/afc_outputs/standard/month_difference_mean.json",
            "w",
        ) as json_file:
            json.dump(month_difference_mean, json_file)
        with open(
            f"{overall_folder}/data_and_models/afc_outputs/standard/month_difference_std.json",
            "w",
        ) as json_file:
            json.dump(month_difference_std, json_file)


def run_measurement_quantile_regressions(df, social):
    columns_of_interest = (
        MEASUREMENT_FREQ_COLS_SOCIAL if social else MEASUREMENT_FREQ_COLS_STANDARD
    )
    Xtrain = df[columns_of_interest]
    ytrain = df[["month_difference"]]

    quantiles = np.arange(0.01, 1.0, 0.01)
    quantiles = [round(x, 2) for x in quantiles]

    gof_results = []
    with ThreadPoolExecutor() as executor:
        measurement_frequency_pre_futures = {
            executor.submit(
                fit_save_measurement_frequency_pre_treatment_models,
                q,
                Xtrain,
                ytrain,
                df,
                social,
            ): q
            for q in quantiles
        }
        for f in as_completed(measurement_frequency_pre_futures):
            q = measurement_frequency_pre_futures[f]
            try:
                result, gof = f.result()
                gof_results.append([q, gof])
                print(f"Measurement model for quantile {q} completed")
            except Exception as e:
                print(f"Measurement model for quantile {q} generated an exception: {e}")

    # as_completed yields in thread-completion order; sort for stable diffs.
    gof_results = (
        pd.DataFrame(gof_results, columns=["quantile", "fit metric"])
        .sort_values("quantile")
        .reset_index(drop=True)
    )
    if social:
        gof_results.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/measurement_gof_results.csv",
            index=False,
        )
    else:
        gof_results.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/measurement_gof_results.csv",
            index=False,
        )


def fit_save_measurement_frequency_pre_treatment_models(q, Xtrain, ytrain, df, social):
    this_model = sm.QuantReg(
        ytrain.astype(float), Xtrain.astype(float), missing="drop"
    ).fit(q=q)
    this_model_1 = sm.QuantReg(
        ytrain.astype(float), df["intercept"].astype(float), missing="drop"
    ).fit(q=q)
    this_fit = quant_goodfit(this_model.resid, this_model_1.resid, q)
    if social:
        this_model.save(
            f"{overall_folder}/data_and_models/afc_models/social/measurement_models_pre_treatment/measure_freq_quantile_{q}_model.pickle",
            remove_data=True,
        )
    else:
        this_model.save(
            f"{overall_folder}/data_and_models/afc_models/standard/measurement_models_pre_treatment/measure_freq_quantile_{q}_model.pickle",
            remove_data=True,
        )

    return this_model, this_fit


def compute_measurement_correlations(df, social):
    quantiles = np.arange(0.01, 1.0, 0.01)
    quantiles = [round(x, 2) for x in quantiles]
    df["year"] = df["last_date"].dt.year

    # predicting measurement frequency quantiles
    if social:
        measurement_frequency_models = [
            sm.load(
                f"{overall_folder}/data_and_models/afc_models/social/measurement_models_pre_treatment/measure_freq_quantile_{q}_model.pickle"
            )
            for q in quantiles
        ]
    else:
        measurement_frequency_models = [
            sm.load(
                f"{overall_folder}/data_and_models/afc_models/standard/measurement_models_pre_treatment/measure_freq_quantile_{q}_model.pickle"
            )
            for q in quantiles
        ]

    df_subset = df[(df["year"] >= 2017) & (df["year"] < 2022)]
    if social:
        measure_X = pd.DataFrame(
            {
                "const": 1,
                "sex": df_subset["female"],
                "SDI": df_subset["SDI"],
                "age": df_subset["age"],
                "last_systolic_bp": df_subset["last_systolic_bp"],
                "last_diastolic_bp": df_subset["last_diastolic_bp"],
            }
        )
    else:
        measure_X = pd.DataFrame(
            {
                "const": 1,
                "sex": df_subset["female"],
                "age": df_subset["age"],
                "last_systolic_bp": df_subset["last_systolic_bp"],
                "last_diastolic_bp": df_subset["last_diastolic_bp"],
            }
        )

    pred_matrix = parallel_predict_models(measurement_frequency_models, measure_X)

    # find the percentile that mostly matches each individual observation
    month_diff_array = df_subset["month_difference"].to_numpy()
    raise_if_nan(month_diff_array, "month_difference")
    percentiles = []
    for i, month_diff in enumerate(month_diff_array):
        diff = np.abs(pred_matrix[i] - month_diff)
        min_idx = np.argmin(diff)
        percentiles.append(quantiles[min_idx])

    df_subset["measurement_freq_percentile"] = percentiles

    pivot_df = df_subset.pivot_table(
        index="person_id",
        columns="year",
        values=["measurement_freq_percentile"],
        aggfunc="mean",
    )

    pivot_df.reset_index(drop=True, inplace=True)
    if social:
        pd.DataFrame(
            pivot_df.corr().values, columns=[df_subset["year"].unique()]
        ).to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/afc_measurement_quantiles.csv",
            index=False,
        )
    else:
        pd.DataFrame(
            pivot_df.corr().values, columns=[df_subset["year"].unique()]
        ).to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/afc_measurement_quantiles.csv",
            index=False,
        )


def main():
    start_time = time.time()
    # parent directory
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)

    parser = ArgumentParser()
    parser.add_argument(
        "-sff", dest="sff", required=True, help='social factors framework ("1"or "0")'
    )
    args = parser.parse_args()
    sff = args.sff
    # if sff == "1", then social factors framework is True
    social = sff == "1"

    if social:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/social/measurement_models_pre_treatment",
            exist_ok=True,
        )
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_outputs/social", exist_ok=True
        )
    else:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/standard/measurement_models_pre_treatment",
            exist_ok=True,
        )
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_outputs/standard", exist_ok=True
        )
    df = read_in_afc_data()
    export_measurement_mean_sd(df, social)
    run_measurement_quantile_regressions(df, social)
    compute_measurement_correlations(df, social)
    end_time = time.time()
    print(f"This took {(end_time - start_time) / 60} minutes")


if __name__ == "__main__":
    main()

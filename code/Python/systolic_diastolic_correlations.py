import os
import pandas as pd
import numpy as np
from bq_config import bq_client, table
import statsmodels.api as sm
import time
from argparse import ArgumentParser
from model_fitting import parallel_predict_models
from paths import overall_folder


def read_in_afc_data():
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
    df = df[df["age"] >= 40]
    df = df.reset_index()
    df["age_rounded"] = df["age"].round(3).astype(float)
    return df


def prepare_data_with_systolic_spline(df, social):
    if social:
        systolic_splines = pd.read_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/systolic_splines.csv"
        )
    else:
        systolic_splines = pd.read_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/systolic_splines.csv"
        )
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    df = convert_nullable_ints(df)
    systolic_splines = convert_nullable_ints(systolic_splines)
    systolic_splines = systolic_splines.sort_values("age")

    df_sorted = df.sort_values(["age_rounded"])
    df_sbp = pd.merge_asof(
        df_sorted,
        systolic_splines,
        left_on="age_rounded",
        right_on="age",
        by="female",
        direction="nearest",
        suffixes=("", "_spline"),
    )
    df_sbp = df_sbp.sort_values("_orig_order").drop(columns="_orig_order")
    # Construct a predictors DataFrame for SBP.
    spline_cols_sbp = list(systolic_splines.columns[2:])
    if social:
        X_sbp = pd.DataFrame(
            {"const": 1, "sex": df_sbp["female"], "SDI": df_sbp["SDI"]}
        )
    else:
        X_sbp = pd.DataFrame({"const": 1, "sex": df_sbp["female"]})

    for col in spline_cols_sbp:
        X_sbp[col] = df_sbp[col]

    return X_sbp


def convert_nullable_ints(dataframe):
    for col in dataframe.columns:
        if pd.api.types.is_integer_dtype(dataframe[col]):
            dataframe[col] = dataframe[col].astype("float64")
    return dataframe


def prepare_data_with_diastolic_spline(df, social):
    if social:
        diastolic_splines = pd.read_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/diastolic_splines.csv"
        )
    else:
        diastolic_splines = pd.read_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/diastolic_splines.csv"
        )
    # --- Vectorized prediction for Diastolic BP quantiles ---
    # Merge with diastolic_splines to get the required spline predictors.
    df = df.copy()
    df = convert_nullable_ints(df)
    diastolic_splines = convert_nullable_ints(diastolic_splines)
    diastolic_splines = diastolic_splines.sort_values("age")

    df["_orig_order"] = np.arange(len(df))
    df_sorted = df.sort_values(["age_rounded"])
    df_dbp = pd.merge_asof(
        df_sorted,
        diastolic_splines,
        left_on="age_rounded",
        right_on="age",
        by="female",
        direction="nearest",
        suffixes=("", "_spline"),
    )
    df_dbp = df_dbp.sort_values("_orig_order").drop(columns="_orig_order")
    spline_cols_dbp = list(diastolic_splines.columns[2:])
    if social:
        X_dbp = pd.DataFrame(
            {"const": 1, "sex": df_dbp["female"], "SDI": df_dbp["SDI"]}
        )
    else:
        X_dbp = pd.DataFrame({"const": 1, "sex": df_dbp["female"]})
    for col in spline_cols_dbp:
        X_dbp[col] = df_dbp[col]

    return X_dbp


def return_sbp_quantile_predictions(sb_quantile_models, quantiles, X_sbp, df):
    # Use parallel processing to get predictions from each SBP quantile model.
    pred_matrix_sbp = parallel_predict_models(sb_quantile_models, X_sbp)
    systolic_bp_array = df["systolic BP"].to_numpy()
    diff_matrix_sbp = np.abs(pred_matrix_sbp - systolic_bp_array[:, np.newaxis])
    min_index_sbp = np.argmin(diff_matrix_sbp, axis=1)
    df["age_percentile"] = np.array(quantiles)[min_index_sbp]
    return df


def return_dbp_quantile_predictions(db_quantile_models, quantiles, X_dbp, df):
    pred_matrix_dbp = parallel_predict_models(db_quantile_models, X_dbp)
    diastolic_bp_array = df["diastolic BP"].to_numpy()
    diff_matrix_dbp = np.abs(pred_matrix_dbp - diastolic_bp_array[:, np.newaxis])
    min_index_dbp = np.argmin(diff_matrix_dbp, axis=1)
    df["age_percentile_dbp"] = np.array(quantiles)[min_index_dbp]
    return df


# Years since each person's first BP measurement, rounded to whole years.
def return_years(df):
    df["years"] = (
        (df["this_date"] - df.groupby("person_id")["this_date"].transform("min"))
        .dt.days.div(365)
        .round()
        .astype(int)
    )
    return df


def compute_correlations(df, age_group):
    # Apply all filters at once
    filtered_df = df[
        (df["age"] >= age_group[0]) & (df["age"] < age_group[1]) & (df["years"] < 4)
    ]

    # Pivot the table
    pivot_df = filtered_df.pivot_table(
        index="person_id",
        columns="years",
        values=["age_percentile", "age_percentile_dbp"],
        aggfunc="mean",
    )

    # Drop the 'person_id' index for correlation computation
    pivot_df.reset_index(drop=True, inplace=True)

    # Compute and return the correlation matrix
    return pivot_df.corr().values


def main():
    start_time = time.time()
    df = read_in_afc_data()

    parser = ArgumentParser()
    parser.add_argument(
        "-sff", dest="sff", required=True, help='social factors framework ("1"or "0")'
    )
    args = parser.parse_args()
    sff = args.sff
    # if sff == "1", then social factors framework is True
    social = sff == "1"

    quantiles = np.arange(0.01, 1.0, 0.01)
    quantiles = [round(x, 2) for x in quantiles]

    if social:
        sb_quantile_models = [
            sm.load(
                f"{overall_folder}/data_and_models/afc_models/social/starting_systolic_bp_regressions/SBP_quantile_{q}_model.pickle"
            )
            for q in quantiles
        ]
        db_quantile_models = [
            sm.load(
                f"{overall_folder}/data_and_models/afc_models/social/starting_diastolic_bp_regressions/DBP_quantile_{q}_model.pickle"
            )
            for q in quantiles
        ]
    else:
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

    X_sbp = prepare_data_with_systolic_spline(df, social)
    df = return_sbp_quantile_predictions(sb_quantile_models, quantiles, X_sbp, df)
    X_dbp = prepare_data_with_diastolic_spline(df, social)
    df = return_dbp_quantile_predictions(db_quantile_models, quantiles, X_dbp, df)

    df = return_years(df)
    # compute age-group-specific correlations for the quantiles
    age_groups = [
        [40, 44],
        [44, 48],
        [48, 52],
        [52, 56],
        [56, 60],
        [60, 64],
        [64, 68],
        [68, 72],
        [72, 76],
        [76, 80],
        [80, 100],
    ]

    correlations = [compute_correlations(df, i) for i in age_groups]

    if social:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_outputs/social/AFC_age_group_correlations",
            exist_ok=True,
        )
    else:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_outputs/standard/AFC_age_group_correlations",
            exist_ok=True,
        )
    # saved these correlations
    for i in range(len(correlations)):
        if social:
            pd.DataFrame(correlations[i]).to_csv(
                f"{overall_folder}/data_and_models/afc_outputs/social/AFC_age_group_correlations/saved_{str(i)}.csv",
                index=False,
            )
        else:
            pd.DataFrame(correlations[i]).to_csv(
                f"{overall_folder}/data_and_models/afc_outputs/standard/AFC_age_group_correlations/saved_{str(i)}.csv",
                index=False,
            )

    end_time = time.time()
    print(f"This took {(end_time - start_time) / 60} minutes")


if __name__ == "__main__":
    main()

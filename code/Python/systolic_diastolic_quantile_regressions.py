import os
import pandas as pd
import numpy as np
from bq_config import bq_client, table
import statsmodels.api as sm
import statsmodels.stats.proportion as smp
from model_fitting import quant_goodfit
from paths import overall_folder
from patsy import dmatrix
import patsy
from sklearn.model_selection import GroupKFold
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import time
from argparse import ArgumentParser


# AFC DATA READ IN
def read_in_afc_data():
    client = bq_client()

    # Reads from a fixed 1M-row snapshot built by `make prime-sample` (see
    # code/SQL/sample_snapshot.sql). The snapshot already filters to SDI is
    # not null and max_number < 1000, and is sampled deterministically so
    # this query is reproducible across pipeline rebuilds.
    query = f"""
    SELECT * from {table("SBP_DBP_values_nonmeds_sample_1m")}
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

    BP_stage_conditions = [
        (df["systolic BP"] >= 140) | (df["diastolic BP"] >= 90),
        (df["systolic BP"] >= 130) | (df["diastolic BP"] >= 80),
        (df["systolic BP"] >= 120) & (df["diastolic BP"] < 80),
    ]
    BP_stage_values = ["S2", "S1", "E"]
    df["BP_stage"] = np.select(BP_stage_conditions, BP_stage_values, default="N")

    BP_stage_dummies = pd.get_dummies(df["BP_stage"], dtype=float)
    df = pd.concat([df, BP_stage_dummies], axis=1)

    df["intercept"] = pd.Series([1 for i in range(len(df))])
    return df


def obtain_age_specific_prev(df, social):
    # Obtain the distribution of BP stages by age in AFC
    ages = np.arange(40, 90)

    grouped = (
        df[df["age"].isin(ages)]
        .groupby("age")
        .agg({"N": "mean", "E": "mean", "S1": "mean", "S2": "mean"})
        .reset_index()
    )

    grouped.columns = ["age", "Normal", "Elevated", "Stage 1", "Stage 2"]

    for stage in ["Normal", "Elevated", "Stage 1", "Stage 2"]:
        grouped[f"{stage} 2.5%"] = 0.0
        grouped[f"{stage} 97.5%"] = 0.0

    for stage in ["N", "E", "S1", "S2"]:
        for age in ages:
            num1 = len(df[(df["age"] == age) & (df["BP_stage"] == stage)])
            num2 = len(df[df["age"] == age])
            ci_low, ci_upp = smp.proportion_confint(
                num1, num2, alpha=0.05, method="normal"
            )
            stage_name = {
                "N": "Normal",
                "E": "Elevated",
                "S1": "Stage 1",
                "S2": "Stage 2",
            }[stage]
            grouped.loc[grouped["age"] == age, f"{stage_name} 2.5%"] = ci_low
            grouped.loc[grouped["age"] == age, f"{stage_name} 97.5%"] = ci_upp

    # export age-specific prevalence of BP stages in PRIME registry
    BP_stage_prevalence = grouped
    if social:
        BP_stage_prevalence.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/AFC_BP_Stage_Prev.csv",
            index=False,
        )
    else:
        BP_stage_prevalence.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/AFC_BP_Stage_Prev.csv",
            index=False,
        )


# Function to compute pinball loss
def pinball_loss(y_true, y_pred, tau):
    return np.mean(np.maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred)))


def evaluate_spline_combination(f, d, df, social, quantile_number=0.50):
    if f < d:
        return None

    spline_basis = dmatrix(
        f"bs(age, df = {f}, degree = {d}, include_intercept = False) * female",
        data=df,
        return_type="dataframe",
    )
    spline_basis = spline_basis.drop(columns=["female"])
    this_df = pd.concat([df, spline_basis], axis=1)
    if social:
        columns_of_interest = ["intercept", "female", "SDI"] + list(
            spline_basis.columns[1:]
        )
    else:
        columns_of_interest = ["intercept", "female"] + list(spline_basis.columns[1:])
    Xtrain = this_df[columns_of_interest]
    ytrain = this_df[["systolic BP"]]
    ytrain2 = this_df[["diastolic BP"]]

    kf = GroupKFold(n_splits=5)
    cv_losses_systolic, cv_losses_diastolic = [], []
    for train_idx, val_idx in kf.split(Xtrain, groups=this_df["person_id"]):
        X_train, X_val = Xtrain.iloc[train_idx], Xtrain.iloc[val_idx]
        y_train, y_val = ytrain.iloc[train_idx], ytrain.iloc[val_idx]
        y_train2, y_val2 = ytrain2.iloc[train_idx], ytrain2.iloc[val_idx]

        systolic_model = sm.QuantReg(
            y_train.astype(float), X_train.astype(float), missing="drop"
        ).fit(q=quantile_number)
        y_pred = systolic_model.predict(X_val)
        cv_losses_systolic.append(
            pinball_loss(y_val["systolic BP"].values, np.array(y_pred), quantile_number)
        )

        diastolic_model = sm.QuantReg(
            y_train2.astype(float), X_train.astype(float), missing="drop"
        ).fit(q=quantile_number)
        y_pred2 = diastolic_model.predict(X_val)
        cv_losses_diastolic.append(
            pinball_loss(
                y_val2["diastolic BP"].values, np.array(y_pred2), quantile_number
            )
        )

    return [f, d, np.mean(cv_losses_systolic), np.mean(cv_losses_diastolic)]


def five_fold_age_spline_CV(df, social):
    # Start cross-validation to select degrees of freedom and dfs for the age spline variable
    df_options = list(range(1, 7))
    degree_options = list(range(1, 5))
    results_arr = []
    with ProcessPoolExecutor() as executor:
        # Submit all valid combinations to the executor
        futures = [
            executor.submit(evaluate_spline_combination, f, d, df, social)
            for f in df_options
            for d in degree_options
            if f >= d
        ]

        # Collect the results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results_arr.append(result)

    # as_completed yields in thread-completion order; sort for stable diffs.
    results_df = (
        pd.DataFrame(
            results_arr, columns=["df", "degree", "systolic_loss", "diastolic_loss"]
        )
        .sort_values(["df", "degree"])
        .reset_index(drop=True)
    )
    if social:
        results_df.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/five_fold_CV_results.csv",
            index=False,
        )
    else:
        results_df.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/five_fold_CV_results.csv",
            index=False,
        )

    df_for_systolic = results_df.iloc[np.argmin(results_df["systolic_loss"])]["df"]
    degrees_for_systolic = results_df.iloc[np.argmin(results_df["systolic_loss"])][
        "degree"
    ]
    df_for_diastolic = results_df.iloc[np.argmin(results_df["diastolic_loss"])]["df"]
    degrees_for_diastolic = results_df.iloc[np.argmin(results_df["diastolic_loss"])][
        "degree"
    ]

    return (
        df_for_systolic,
        degrees_for_systolic,
        df_for_diastolic,
        degrees_for_diastolic,
    )


def save_sbp_spline(df, df_for_systolic, degrees_for_systolic, social):
    spline_basis = dmatrix(
        f"bs(age, df = {int(df_for_systolic)}, degree = {int(degrees_for_systolic)}, include_intercept = False) * female",
        data=df,
        return_type="dataframe",
    )

    # save the b-spline for each age and sex for systolic blood pressure
    age_systolic_splines = []
    for a in np.arange(40, 101, 1 / 12):
        for f in range(2):
            x = np.asarray(
                patsy.build_design_matrices(
                    [spline_basis.design_info], {"age": a, "female": f}
                )[0]
            ).tolist()[0]
            age_systolic_splines.append([round(a, 3), f] + x)
    age_systolic_spline_columns = ["age", "female"] + list(spline_basis.columns)
    age_systolic_splines = pd.DataFrame(
        age_systolic_splines, columns=age_systolic_spline_columns
    )
    age_systolic_splines = age_systolic_splines.drop(columns=["Intercept"])
    age_systolic_splines = age_systolic_splines.loc[
        :, ~age_systolic_splines.columns.duplicated()
    ]
    if social:
        age_systolic_splines.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/systolic_splines.csv",
            index=False,
        )
    else:
        age_systolic_splines.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/systolic_splines.csv",
            index=False,
        )

    return spline_basis


def save_dbp_spline(df, df_for_diastolic, degrees_for_diastolic, social):
    spline_basis = dmatrix(
        f"bs(age, df = {int(df_for_diastolic)}, degree = {int(degrees_for_diastolic)}, include_intercept = False) * female",
        data=df,
        return_type="dataframe",
    )
    # save the b-spline for each age and sex for diastolic blood pressure
    age_diastolic_splines = []
    for a in np.arange(40, 101, 1 / 12):
        for f in range(2):
            x = np.asarray(
                patsy.build_design_matrices(
                    [spline_basis.design_info], {"age": a, "female": f}
                )[0]
            ).tolist()[0]
            age_diastolic_splines.append([round(a, 3), f] + x)
    age_diastolic_spline_columns = ["age", "female"] + list(spline_basis.columns)
    age_diastolic_splines = pd.DataFrame(
        age_diastolic_splines, columns=age_diastolic_spline_columns
    )
    age_diastolic_splines = age_diastolic_splines.drop(columns=["Intercept"])
    age_diastolic_splines = age_diastolic_splines.loc[
        :, ~age_diastolic_splines.columns.duplicated()
    ]
    if social:
        age_diastolic_splines.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/diastolic_splines.csv",
            index=False,
        )
    else:
        age_diastolic_splines.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/diastolic_splines.csv",
            index=False,
        )
    return spline_basis


def fit_and_save_sbp_model(q, Xtrain, ytrain, this_df, social):
    this_model = sm.QuantReg(
        ytrain.astype(float), Xtrain.astype(float), missing="drop"
    ).fit(q=q)
    this_model_1 = sm.QuantReg(
        ytrain.astype(float), this_df["intercept"].astype(float), missing="drop"
    ).fit(q=q)
    this_fit = quant_goodfit(this_model.resid, this_model_1.resid, q)
    if social:
        this_model.save(
            f"{overall_folder}/data_and_models/afc_models/social/starting_systolic_bp_regressions/SBP_quantile_{q}_model.pickle",
            remove_data=True,
        )
    else:
        this_model.save(
            f"{overall_folder}/data_and_models/afc_models/standard/starting_systolic_bp_regressions/SBP_quantile_{q}_model.pickle",
            remove_data=True,
        )

    return this_model, this_fit


def fit_and_save_dbp_model(q, Xtrain2, ytrain2, this_df2, social):
    this_model = sm.QuantReg(
        ytrain2.astype(float), Xtrain2.astype(float), missing="drop"
    ).fit(q=q)
    this_model_1 = sm.QuantReg(
        ytrain2.astype(float), this_df2["intercept"].astype(float), missing="drop"
    ).fit(q=q)
    this_fit = quant_goodfit(this_model.resid, this_model_1.resid, q)
    if social:
        this_model.save(
            f"{overall_folder}/data_and_models/afc_models/social/starting_diastolic_bp_regressions/DBP_quantile_{q}_model.pickle",
            remove_data=True,
        )
    else:
        this_model.save(
            f"{overall_folder}/data_and_models/afc_models/standard/starting_diastolic_bp_regressions/DBP_quantile_{q}_model.pickle",
            remove_data=True,
        )

    return this_model, this_fit


def run_bp_quantile_regressions(spline_basis_sbp, spline_basis_dbp, df, social):
    quantiles = np.arange(0.01, 1.0, 0.01)
    quantiles = [round(x, 2) for x in quantiles]

    # spline to use in the sbp quantile regressions
    spline_basis_sbp = spline_basis_sbp.drop(columns=["female"])
    this_df = pd.concat([df, spline_basis_sbp], axis=1)
    if social:
        columns_of_interest = ["intercept", "female", "SDI"] + list(
            spline_basis_sbp.columns[1:]
        )
    else:
        columns_of_interest = ["intercept", "female"] + list(
            spline_basis_sbp.columns[1:]
        )
    Xtrain = this_df[columns_of_interest]
    ytrain = this_df[["systolic BP"]]

    # dbp quantile regressions
    spline_basis_dbp = spline_basis_dbp.drop(columns=["female"])
    this_df2 = pd.concat([df, spline_basis_dbp], axis=1)
    if social:
        columns_of_interest2 = ["intercept", "female", "SDI"] + list(
            spline_basis_dbp.columns[1:]
        )
    else:
        columns_of_interest2 = ["intercept", "female"] + list(
            spline_basis_dbp.columns[1:]
        )
    Xtrain2 = this_df2[columns_of_interest2]
    ytrain2 = this_df2[["diastolic BP"]]

    gof_results = []
    with ThreadPoolExecutor() as executor:
        sbp_futures = {
            executor.submit(
                fit_and_save_sbp_model, q, Xtrain, ytrain, this_df, social
            ): q
            for q in quantiles
        }
        dbp_futures = {
            executor.submit(
                fit_and_save_dbp_model, q, Xtrain2, ytrain2, this_df2, social
            ): q
            for q in quantiles
        }

        # Optionally, retrieve results or handle exceptions
        for f in as_completed(sbp_futures):
            q = sbp_futures[f]
            try:
                result, gof = f.result()
                gof_results.append([q, "sys", gof])
                print(f"SBP model for quantile {q} completed.")
            except Exception as e:
                print(f"SBP model for quantile {q} generated an exception: {e}")

        for f in as_completed(dbp_futures):
            q = dbp_futures[f]
            try:
                result, gof = f.result()
                gof_results.append([q, "dia", gof])
                print(f"DBP model for quantile {q} completed.")
            except Exception as e:
                print(f"DBP model for quantile {q} generated an exception: {e}")

    # as_completed yields in thread-completion order; sort for stable diffs.
    gof_results = (
        pd.DataFrame(gof_results, columns=["quantile", "outcome", "fit metric"])
        .sort_values(["outcome", "quantile"])
        .reset_index(drop=True)
    )
    if social:
        gof_results.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/social/systolic_diastolic_gof_results.csv",
            index=False,
        )
    else:
        gof_results.to_csv(
            f"{overall_folder}/data_and_models/afc_outputs/standard/systolic_diastolic_gof_results.csv",
            index=False,
        )


def main():
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument(
        "-sff", dest="sff", required=True, help='social factors framework ("1"or "0")'
    )
    args = parser.parse_args()
    sff = args.sff
    # if sff == "1", then social factors framework is True
    social = sff == "1"

    # parent directory
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)

    # make sure the directories for saving the AFC models exist
    os.makedirs(f"{overall_folder}/data_and_models/afc_models", exist_ok=True)
    # existing folders
    if social:
        ##AFC MODEL FOLDERS
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/social/", exist_ok=True
        )
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/social/starting_systolic_bp_regressions",
            exist_ok=True,
        )
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/social/starting_diastolic_bp_regressions",
            exist_ok=True,
        )
        #
    else:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/standard", exist_ok=True
        )
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/standard/starting_systolic_bp_regressions",
            exist_ok=True,
        )
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/standard/starting_diastolic_bp_regressions",
            exist_ok=True,
        )

    # make sure the directories for saving AFC model outputs exists
    os.makedirs(f"{overall_folder}/data_and_models/afc_outputs", exist_ok=True)
    if social:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_outputs/social/", exist_ok=True
        )
    else:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_outputs/standard/", exist_ok=True
        )

    df = read_in_afc_data()

    ##MARIKA'S TEST DATA
    # df = simulate_afc_data()

    obtain_age_specific_prev(df, social)

    df_for_systolic, degrees_for_systolic, df_for_diastolic, degrees_for_diastolic = (
        five_fold_age_spline_CV(df, social)
    )

    spline_basis_sbp = save_sbp_spline(
        df, df_for_systolic, degrees_for_systolic, social
    )
    spline_basis_dbp = save_dbp_spline(
        df, df_for_diastolic, degrees_for_diastolic, social
    )

    run_bp_quantile_regressions(spline_basis_sbp, spline_basis_dbp, df, social)

    end_time = time.time()
    print(f"This took {(end_time - start_time) / 60} minutes")


if __name__ == "__main__":
    main()

import pandas as pd
from bq_config import BILLING_PROJECT, bq_client, table
import pickle
import numpy as np
import os
from statsmodels.formula.api import logit
from argparse import ArgumentParser


def read_in_afc_data():
    client = bq_client(BILLING_PROJECT)

    # Reads from the stage_1_2_obs_with_max_prev_num built table
    # (see code/SQL/hypertension_sql_queries.sql); the post-2018-01-01 filter,
    # non-null SDI_CT, and per-person max(prev_num) < 10 cap are baked in there.
    query = f"""
    select * from {table("stage_1_2_obs_with_max_prev_num")}
    """

    query_job = client.query(query)
    df = query_job.to_dataframe()

    df["female"] = np.where(df["gender"] == "FEMALE", 1, 0)

    df["first_exposure"] = pd.to_datetime(df["first_exposure"])
    df["measurement_datetime"] = pd.to_datetime(df["measurement_datetime"])

    df["treated_now"] = (
        df["first_exposure"].notnull()
        & ((df["first_exposure"] - df["measurement_datetime"]).dt.days <= 31)
    ).astype(int)

    return df


def fit_treatment_logit_model(df, social):
    if social:
        logit_model = logit(
            "treated_now ~ age + female + SDI_CT + systolic_bp + diastolic_bp", df
        ).fit()
    else:
        logit_model = logit(
            "treated_now ~ age + female + systolic_bp + diastolic_bp", df
        ).fit()

    return logit_model


def main():
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

    df = read_in_afc_data()
    logit_model = fit_treatment_logit_model(df, social)
    if social:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/social/treatment_models",
            exist_ok=True,
        )
        with open(
            f"{overall_folder}/data_and_models/afc_models/social/treatment_models/logistic_treatment_model.pickle",
            "wb",
        ) as f:
            pickle.dump(logit_model.params, f)
    else:
        os.makedirs(
            f"{overall_folder}/data_and_models/afc_models/standard/treatment_models",
            exist_ok=True,
        )
        with open(
            f"{overall_folder}/data_and_models/afc_models/standard/treatment_models/logistic_treatment_model.pickle",
            "wb",
        ) as f:
            pickle.dump(logit_model.params, f)


if __name__ == "__main__":
    main()

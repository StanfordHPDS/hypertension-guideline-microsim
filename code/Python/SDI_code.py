import os
import numpy as np
import pandas as pd
from bq_config import bq_client, table
from argparse import ArgumentParser
from sdi import return_SDI
import time


def main():
    # parent directory
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)

    start_time = time.time()
    client = bq_client()

    query = f"""
    SELECT *
    from {table("CT_SDI")}
    where SDI_CT is not NULL
    """

    query_job = client.query(query)
    SDI_AFC_df = query_job.to_dataframe()

    parser = ArgumentParser()
    parser.add_argument(
        "-d", dest="NHANES_file", required=True, help="NHANES file name"
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Master seed for the SDI imputation RNG (default: 42).",
    )
    args = parser.parse_args()
    NHANES_file = args.NHANES_file
    rng = np.random.default_rng(args.seed)

    NHANES_cohort = pd.read_csv(
        f"{overall_folder}/data_and_models/nhanes_data_cohorts/{NHANES_file}.csv"
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
    ]

    ##replace with SDI_CT_2023.csv
    SDI_CT_df = pd.read_csv(
        f"{overall_folder}/data_and_models/nhanes_inputs/SDI_CT.csv",
        dtype={"GEOID": str},
    )

    SDI_CT_df["prev"] = SDI_CT_df["B01001_001"] / SDI_CT_df["B01001_001"].sum()
    SDI_CT_df = SDI_CT_df.rename(
        columns={
            "GEOID": "CENSUSTRACT_FIPS",
            "POV_P": "pct_Poverty_LT100",
            "NOHSDP_P": "pct_Education_LT12years",
            "NONEMP_P": "pct_NonEmployed",
            "RENT_P": "pct_HH_Renter_Occupied",
            "CROWD_P": "pct_HH_Crowding",
            "SNGPNT_P": "pct_Single_Parent_Fam",
        }
    )

    area_data = SDI_CT_df[
        [
            "CENSUSTRACT_FIPS",
            "pct_Poverty_LT100",
            "pct_Single_Parent_Fam",
            "pct_Education_LT12years",
            "pct_NonEmployed",
            "pct_HH_Renter_Occupied",
            "pct_HH_Crowding",
            "prev",
        ]
    ]

    # new method to assign SDI to NHANES cohort
    NHANES_cohort["SDI"] = NHANES_cohort.apply(
        return_SDI, axis=1, args=(area_data, SDI_AFC_df, rng)
    )

    os.makedirs(f"{overall_folder}/data_and_models/nhanes_data_cohorts", exist_ok=True)
    NHANES_cohort.to_csv(
        f"{overall_folder}/data_and_models/nhanes_data_cohorts/{NHANES_file}_sdi.csv",
        index=False,
    )

    end_time = time.time()
    print(f"This took {(end_time - start_time) / 60} minutes")


if __name__ == "__main__":
    main()

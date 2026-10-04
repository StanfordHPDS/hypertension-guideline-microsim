import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
import dask.dataframe as dd
from argparse import ArgumentParser
import os
from bp_generation import (
    generate_hypertension_stage,
    generate_new_DBP_value,
    generate_new_SBP_value,
)
from mortality import create_death_adjustment_table, load_all_life_tables
from numeric import convert_to_prob, convert_to_rate

# start to run the patient cohort
start_time = time.time()

# parent directory
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
overall_folder = os.path.dirname(parent_directory)

life_table_dict = load_all_life_tables(overall_folder)

CYCLE_LENGTH = 1 / 12.0

quantiles = np.arange(0.01, 1.0, 0.01)
quantiles = [round(x, 2) for x in quantiles]
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

systolic_splines = pd.read_csv(
    f"{overall_folder}/data_and_models/afc_outputs/standard/systolic_splines.csv"
)
diastolic_splines = pd.read_csv(
    f"{overall_folder}/data_and_models/afc_outputs/standard/diastolic_splines.csv"
)


# function to generate the next systolic and diastolic blood pressure value and corresponding
# BP stage
def generate_next_BP_stage(
    this_quantile,
    this_quantile_dbp,
    current_state_DNH,
    age,
    sex,
):
    if current_state_DNH == "A":
        new_SBP = generate_new_SBP_value(
            sb_quantile_models, systolic_splines, this_quantile, age, sex
        )
        new_DBP = generate_new_DBP_value(
            db_quantile_models, diastolic_splines, this_quantile_dbp, age, sex
        )

        return new_SBP, new_DBP, generate_hypertension_stage(new_SBP, new_DBP)
    else:
        return np.nan, np.nan, "D"


DBP_values = [65, 72, 76, 82, 90]
DBP_HR_values = [1, 1.8, 2, 2, 3]
SBP_values = [115, 124, 130, 140, 155]
SBP_HR_values = [1, 2, 2, 2.4, 4]

hr_adjust_table = create_death_adjustment_table()


# function to generate the probability of transitions within the disease natural history (either Alive or Dead)
def generate_transitions_DNH(
    current_state_DNH, current_DBP_val, current_SBP_val, age, sex, race
):
    transition_vec = {"A": [0, 0], "D": [0, 1]}  # Default transition for age >= 100

    if age >= 100:
        transition_vec["A"] = [0, 1]
        return transition_vec[current_state_DNH]

    # Select the appropriate life table based on race and sex
    this_life_table = life_table_dict.get((race, sex))

    pHD = this_life_table[this_life_table["age"] == int(age)]["qx_cycle"].iloc[0]

    rHD = convert_to_rate(pHD)
    # Calculate hazard ratios
    DBP_HR = np.interp(current_DBP_val, DBP_values, DBP_HR_values, left=1, right=3)
    SBP_HR = np.interp(current_SBP_val, SBP_values, SBP_HR_values, left=1, right=4)
    avg_HR = np.mean([DBP_HR, SBP_HR])

    HR_adjustment = hr_adjust_table[hr_adjust_table["age"] == int(age)]["avg_HR"].iloc[
        0
    ]

    rSD = (rHD * avg_HR) / HR_adjustment
    pSD = convert_to_prob(rSD)

    transition_vec["A"] = [1 - pSD, pSD]
    return transition_vec[current_state_DNH]


def sample_quantile_change_schedule(
    N,
    cycles,
    random_seeds,
    mean_years_between_changes=5,
    min_gap_months=12,
):
    mean_months = mean_years_between_changes * 12

    schedules = []

    for i in range(N):
        random_seed = random_seeds.iloc[i]
        rng = np.random.default_rng(random_seed)
        t = 0
        person_schedule = []

        while True:
            gap = int(round(rng.exponential(mean_months)))
            gap = max(gap, min_gap_months)  # avoid unrealistically frequent changes
            t += gap

            if t > cycles:
                break

            person_schedule.append(t)

        schedules.append(person_schedule)

    return schedules


parser = ArgumentParser()
parser.add_argument("-f", dest="folder", required=True, help="cohort path")
parser.add_argument("-n", dest="cohort_num", required=True, help="cohort path")

args = parser.parse_args()
folder = args.folder
cohort_num = args.cohort_num

print(
    f"{overall_folder}/validation_results/{folder}/saved_results/BP_state_{cohort_num}/"
)

sampled_systolic_quantiles = pd.read_csv(
    f"{overall_folder}/validation_results/{folder}/sampled_systolic_quantiles_{cohort_num}.csv"
)
sampled_systolic_quantiles = np.array(sampled_systolic_quantiles)
sampled_diastolic_quantiles = pd.read_csv(
    f"{overall_folder}/validation_results/{folder}/sampled_diastolic_quantiles_{cohort_num}.csv"
)
sampled_diastolic_quantiles = np.array(sampled_diastolic_quantiles)

sampled_cohort = pd.read_csv(
    f"{overall_folder}/validation_results/{folder}/sampled_cohort_{cohort_num}.csv"
)

# Initalize the simulation cohort
N = len(sampled_cohort)  # individuals
cycles = int(60 / CYCLE_LENGTH) + 5

DNH_states = ["A", "D"]
BP_states = ["N", "E", "S1", "S2"]
DNH_state_trace = np.full((N, cycles + 1), "ToFill", dtype=object)
BP_state_trace = np.full((N, cycles + 1), "ToFill", dtype=object)
SBP_state_trace = np.zeros((N, cycles + 1), dtype=float)
DBP_state_trace = np.zeros((N, cycles + 1), dtype=float)

DNH_state_trace[:, 0] = "A"
BP_state_trace[:, 0] = sampled_cohort["predicted_BP_stage"]
SBP_state_trace[:, 0] = sampled_cohort["predicted_systolic_bp"]
DBP_state_trace[:, 0] = sampled_cohort["predicted_diastolic_bp"]

race_group_conditions = [
    (sampled_cohort["black"] == 1),
    (sampled_cohort["white"] == 1),
    (sampled_cohort["hispanic"] == 1),
]
race_options = ["NHB", "NHW", "H"]
race_values = np.select(race_group_conditions, race_options, default="O")

age_values = [40] * N
sex_values = sampled_cohort["female"]
this_insurance = "Y"
insurance_values = [this_insurance for x in range(N)]
starting_systolic_quantiles = sampled_cohort["age_systolic_percentile"]
starting_diastolic_quantiles = sampled_cohort["age_diastolic_percentile"]

population_df = pd.DataFrame(list(range(0, N)), columns=["id"])
population_df["starting_age"] = pd.Series(age_values)
population_df["race"] = pd.Series(race_values)
population_df["sex"] = pd.Series(sex_values)
population_df["insurance"] = pd.Series(insurance_values)

quantile_change_schedules = sample_quantile_change_schedule(
    N=len(sampled_cohort),
    cycles=cycles,
    random_seeds=sampled_cohort["random_seed"],
    mean_years_between_changes=8,
    min_gap_months=12,
)

for i in range(N):
    # count = 1

    person_schedule = set(quantile_change_schedules[i])
    count = 0
    # Max available future quantile columns for this person
    max_count_sbp = len(sampled_systolic_quantiles[i])
    max_count_dbp = len(sampled_diastolic_quantiles[i])

    np.random.seed(sampled_cohort["random_seed"].iloc[i])
    for t in range(cycles):
        ##OLD VERSION
        # if we are in the first year (12 months), we are going to use the starting systolic and diastolic blood pressure quantiles
        # if t < 12:
        #     this_quantile_sbp = starting_systolic_quantiles[i]
        #     this_quantile_dbp = starting_diastolic_quantiles[i]

        # # if the number is divisible by 12 (every year), we are going to swtich the systolic and diastolic blood pressure quantiles
        # elif t > 0 and t % 12 == 0:
        #     this_quantile_sbp = sampled_systolic_quantiles[i][count]
        #     this_quantile_dbp = sampled_diastolic_quantiles[i][count]
        #     count = count + 1

        # else:
        #     this_quantile_sbp = sampled_systolic_quantiles[i][count]
        #     this_quantile_dbp = sampled_diastolic_quantiles[i][count]

        ##NEW VERSION
        # Update quantiles only when this person hits one of their scheduled change months
        if t == 0:
            this_quantile_sbp = min(max(starting_systolic_quantiles[i], 0.01), 0.99)
            this_quantile_dbp = min(max(starting_diastolic_quantiles[i], 0.01), 0.99)

        if t > 0 and t in person_schedule:
            if count < max_count_sbp:
                this_quantile_sbp = min(
                    max(sampled_systolic_quantiles[i][int(t * CYCLE_LENGTH)], 0.01),
                    0.99,
                )
            if count < max_count_dbp:
                this_quantile_dbp = min(
                    max(sampled_diastolic_quantiles[i][int(t * CYCLE_LENGTH)], 0.01),
                    0.99,
                )

            count += 1

        (
            SBP_state_trace[i, t + 1],
            DBP_state_trace[i, t + 1],
            BP_state_trace[i, t + 1],
        ) = generate_next_BP_stage(
            quantiles.index(this_quantile_sbp),
            quantiles.index(this_quantile_dbp),
            DNH_state_trace[i, t],
            age_values[i],
            sex_values[i],
        )

        this_transition_DNH = generate_transitions_DNH(
            DNH_state_trace[i, t],
            SBP_state_trace[i, t],
            DBP_state_trace[i, t],
            age_values[i],
            sex_values[i],
            race_values[i],
        )
        DNH_state_trace[i, t + 1] = np.random.choice(
            DNH_states,
            size=1,
            p=this_transition_DNH,
        )[0]

        age_values[i] = age_values[i] + CYCLE_LENGTH

# first index of death per individual
death_indices = np.argmax(DNH_state_trace == "D", axis=1)
# whether death occured
death_occurred = np.any(DNH_state_trace == "D", axis=1)
# time until death
years_to_death = np.where(death_occurred, death_indices * CYCLE_LENGTH, 0)
death_age = population_df["starting_age"].values + years_to_death

# Calculate first index for S1 and S2
sick_indices_S1 = np.argmax(BP_state_trace == "S1", axis=1)
sick_indices_S2 = np.argmax(BP_state_trace == "S2", axis=1)
sick_occurred_S1 = np.any(BP_state_trace == "S1", axis=1)
sick_occurred_S2 = np.any(BP_state_trace == "S2", axis=1)

# first index for sick (either S1 or S2)
first_sick = np.full(DNH_state_trace.shape[0], np.inf)
first_sick[sick_occurred_S1] = sick_indices_S1[sick_occurred_S1]
first_sick[sick_occurred_S2] = np.minimum(
    first_sick[sick_occurred_S2], sick_indices_S2[sick_occurred_S2]
)
first_sick[first_sick == np.inf] = 0  # No sickness recorded

first_sick *= CYCLE_LENGTH
sick_age = population_df["starting_age"].values + first_sick

# number of cycles with either S1 or S2
sick_cycles = np.sum((BP_state_trace == "S1") | (BP_state_trace == "S2"), axis=1)
years_sick = sick_cycles * CYCLE_LENGTH
was_sick = years_sick > 0

population_df = dd.from_array(population_df.values, columns=population_df.columns)
columns_trace2 = ["Year " + str(x) for x in range(0, cycles + 1)]
state_trace_df = dd.from_array(np.array(DNH_state_trace), columns=columns_trace2)
columns_trace2 = ["BPStageYear " + str(x) for x in range(0, cycles + 1)]
BP_state_trace_df = dd.from_array(np.array(BP_state_trace), columns=columns_trace2)
columns_trace2 = ["SBPValYear " + str(x) for x in range(0, cycles + 1)]
SBP_val_trace_df = dd.from_array(np.array(SBP_state_trace), columns=columns_trace2)
columns_trace2 = ["DPValYear " + str(x) for x in range(0, cycles + 1)]
DBP_val_trace_df = dd.from_array(np.array(DBP_state_trace), columns=columns_trace2)

statistics_df = dd.from_array(
    np.stack(
        (
            years_to_death,
            death_occurred,
            death_age,
            years_sick,
            was_sick,
            sick_age,
        ),
        axis=-1,
    ),
    columns=[
        "years_to_death",
        "death_occurred",
        "death_age",
        "years_sick",
        "was_sick",
        "sick_age",
    ],
)

os.makedirs(
    f"{overall_folder}/validation_results/{folder}/saved_results/BP_state_{cohort_num}/",
    exist_ok=True,
)
os.makedirs(
    f"{overall_folder}/validation_results/{folder}/saved_results/statistics_{cohort_num}/",
    exist_ok=True,
)
BP_state_trace_df.to_parquet(
    f"{overall_folder}/validation_results/{folder}/saved_results/BP_state_{cohort_num}/"
)
statistics_df.to_parquet(
    f"{overall_folder}/validation_results/{folder}/saved_results/statistics_{cohort_num}/"
)


end_time = time.time()
print(f"This took {(end_time - start_time) / 60} minutes")

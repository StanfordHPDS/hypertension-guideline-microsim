import numpy as np
import pandas as pd
import pickle
import time
from argparse import ArgumentParser
import os
from demographics import compute_weighted_htn_prevalence_by_age
from numeric import CYCLE_LENGTH
from plotting import (
    plot_cum_incidence_control_hypertension,
    plot_cum_treatment,
    plot_cum_treatment_both_groups,
    plot_cumulative_incidence_first_htn,
    plot_DNH_trace,
    plot_hyp_incidence_by_group,
    plot_OHS_calibration_targets,
    run_HS_state_graph,
)
from stats import (
    create_overall_results,
    create_results_by_group,
    create_results_multiple_groups,
)
from vectorized_hypertension_simulation_functions_sff import (
    generate_next_BP_stage_sff,
    generate_transitions_DNH_sff,
    generate_transitions_HS_sff,
    quantiles,
    sample_quantile_change_schedule,
)
import multiprocessing as mp

# parent directory
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
overall_folder = os.path.dirname(parent_directory)

parser = ArgumentParser()
parser.add_argument("-f", dest="folder", required=True, help="folder path")
parser.add_argument(
    "-a", dest="assumption", required=True, help="heterogeneity assumption"
)
parser.add_argument(
    "-ra", dest="race_adjustment", required=True, help="race adjustment"
)

args = parser.parse_args()

##WHAT ARE WE GOING TO RUN
folder = args.folder
hetero_assumption = args.assumption  # either hetero or nohetero
race_adjust = args.race_adjustment  # either RA or NR

hetero_effect = hetero_assumption == "hetero"

assumption_path = hetero_assumption + "_" + race_adjust


##Read in the cohort and the sampled systolic and diastolic quantiles
sampled_systolic_quantiles = pd.read_csv(
    f"{overall_folder}/results/{folder}/cohort_files/social/sampled_systolic_quantiles.csv"
)
sampled_systolic_quantiles = np.array(sampled_systolic_quantiles)
sampled_diastolic_quantiles = pd.read_csv(
    f"{overall_folder}/results/{folder}/cohort_files/social/sampled_diastolic_quantiles.csv"
)
sampled_diastolic_quantiles = np.array(sampled_diastolic_quantiles)

sampled_cohort = pd.read_csv(
    f"{overall_folder}/results/{folder}/cohort_files/social/sampled_cohort.csv"
)

# set-up the population of people
N = len(sampled_cohort)  # individuals
np.random.seed(123)

starting_age = 40
cycles = int(60 / CYCLE_LENGTH) + 5

DNH_states = ["A", "D"]
HS_states = ["OHS", "IHS", "DT", "DUT"]
BP_states = ["N", "E", "S1", "S2"]

starting_hs_state = np.where(sampled_cohort["place"] == 1, "IHS", "OHS")
starting_bp_state = sampled_cohort["predicted_BP_stage"].values
starting_systolic_bp_values = sampled_cohort["predicted_systolic_bp"].values
starting_diastolic_bp_values = sampled_cohort["predicted_diastolic_bp"].values

sex_values = sampled_cohort["female"]

treatment_options = ["TD", "ACE"]  # Example options
treatment_weights = [0.40, 0.60]  # Example weights
random_draws = np.random.rand(N)
treatment_values = np.where(
    random_draws < treatment_weights[0], treatment_options[0], treatment_options[1]
)

# Initialize race and insurance values
race_values = np.where(sampled_cohort["black"] == 1, "NHB", "NHW")
insurance_values = np.where(sampled_cohort["insurance"] == 0, "N", "Y")

# Adjust treatment values for black individuals if race_adjust is 'RA'
if race_adjust == "RA":
    treatment_values = np.where(race_values == "NHB", "TD", treatment_values)

SDI_values = sampled_cohort["SDI"]
starting_systolic_quantiles = sampled_cohort["age_systolic_percentile"].values
starting_diastolic_quantiles = sampled_cohort["age_diastolic_percentile"].values

# setting the initial measurement window
measurement_interval = sampled_cohort["predicted_month_difference"].values
measurement_quantile = sampled_cohort["measurement_freq_percentile"].values

quantile_change_schedules = sample_quantile_change_schedule(
    N=len(sampled_cohort),
    cycles=cycles,
    random_seeds=sampled_cohort["random_seed"],
    mean_years_between_changes=8,
    min_gap_months=12,
)


def simulate_individual_social_factors(i):
    np.random.seed(sampled_cohort["random_seed"].iloc[i])
    age = starting_age
    # Initialize trace arrays for this individual
    hs_trace = np.empty(cycles + 1, dtype=object)
    bp_trace = np.empty(cycles + 1, dtype=object)
    sbp_trace = np.empty(cycles + 1, dtype=object)
    dbp_trace = np.empty(cycles + 1, dtype=object)

    # Fill in initial state
    hs_trace[0] = starting_hs_state[i]
    bp_trace[0] = starting_bp_state[i]
    sbp_trace[0] = starting_systolic_bp_values[i]
    dbp_trace[0] = starting_diastolic_bp_values[i]

    measurement_interval_local = measurement_interval[i]
    person_schedule = set(quantile_change_schedules[i])

    # This tracks which future column to use next
    count = 0
    time_treated = 1 if hs_trace[0] == "DT" else 0

    # Max available future quantile columns for this person
    max_count_sbp = len(sampled_systolic_quantiles[i])
    max_count_dbp = len(sampled_diastolic_quantiles[i])

    for t in range(cycles):
        (
            this_transition_HS,
            measurement_interval_local,
        ) = generate_transitions_HS_sff(
            hs_trace[t],
            bp_trace[t],
            insurance_values[i],
            age,
            sex_values[i],
            SDI_values[i],
            sbp_trace[t],
            dbp_trace[t],
            treatment_values[i],
            measurement_interval_local,
            measurement_quantile[i],
            hetero_effect,
            race_values[i],
        )

        hs_trace[t + 1] = np.random.choice(HS_states, size=1, p=this_transition_HS)[0]
        if hs_trace[t + 1] == "DT":
            time_treated += 1

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

        ## OLD VERSION
        # if t < 12:
        #     this_quantile_sbp = min(max(starting_systolic_quantiles[i], 0.01), 0.99)
        #     this_quantile_dbp = min(max(starting_diastolic_quantiles[i], 0.01), 0.99)

        # # afterwards, we are switching to new systolic and diastolic quantiles every 12 months
        # elif t > 0 and t % 12 == 0:
        #     this_quantile_sbp = min(
        #         max(sampled_systolic_quantiles[i][count], 0.01), 0.99
        #     )
        #     this_quantile_dbp = min(
        #         max(sampled_diastolic_quantiles[i][count], 0.01), 0.99
        #     )
        #     count = count + 1

        # else:
        #     this_quantile_sbp = min(
        #         max(sampled_systolic_quantiles[i][count], 0.01), 0.99
        #     )
        #     this_quantile_dbp = min(
        #         max(sampled_diastolic_quantiles[i][count], 0.01), 0.99
        #     )

        (
            sbp_trace[t + 1],
            dbp_trace[t + 1],
            bp_trace[t + 1],
        ) = generate_next_BP_stage_sff(
            quantiles.index(this_quantile_sbp),
            quantiles.index(this_quantile_dbp),
            hs_trace[t],
            time_treated,
            bp_trace[t],
            age,
            sex_values[i],
            SDI_values[i],
            race_values[i],
            treatment_values[i],
            hetero_effect,
        )

        this_transition_DNH = generate_transitions_DNH_sff(
            bp_trace[t],
            dbp_trace[t],
            sbp_trace[t],
            age,
            sex_values[i],
            race_values[i],
            insurance_values[i],
        )
        alive = np.random.choice(DNH_states, size=1, p=this_transition_DNH)[0]
        if alive == "D":
            bp_trace[t + 1] = "D"

        age += CYCLE_LENGTH

    death_idx = np.where(bp_trace == "D")[0]
    years_to_death = death_idx[0] * CYCLE_LENGTH if len(death_idx) > 0 else 0
    death_age = starting_age + years_to_death

    sick_idx = np.where((bp_trace == "S1") | (bp_trace == "S2"))[0]
    alive_idx = np.where(bp_trace != "D")[0]
    treat_idx = np.where(hs_trace == "DT")[0]
    overlap = np.intersect1d(sick_idx, treat_idx)
    overlap2 = np.intersect1d(alive_idx, treat_idx)

    was_sick = int(len(sick_idx) > 0)
    was_treated = int(len(treat_idx) > 0)
    years_sick = len(sick_idx) * CYCLE_LENGTH
    years_treated = len(overlap2) * CYCLE_LENGTH
    years_sick_treated = len(overlap) * CYCLE_LENGTH

    if "DT" in hs_trace:
        years_to_treated = np.where(hs_trace == "DT")[0][0] * CYCLE_LENGTH
    else:
        years_to_treated = 0

    if len(sick_idx) > 0:
        first_sick = min(sick_idx) * CYCLE_LENGTH
    else:
        first_sick = 0

    return i, {
        "years_to_death": years_to_death,
        "death_age": death_age,
        "years_sick": years_sick,
        "years_treated": years_treated,
        "years_sick_treated": years_sick_treated,
        "years_to_treated": years_to_treated,
        "was_sick": was_sick,
        "was_treated": was_treated,
        "sick_age": starting_age + first_sick,
        "treated_age": starting_age + years_to_treated,
        "hs_trace": hs_trace,
        "bp_trace": bp_trace,
        "sbp_trace": sbp_trace,
        "dbp_trace": dbp_trace,
    }


def run_parallel_simulation(N):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pairs = list(
            pool.imap_unordered(
                simulate_individual_social_factors, range(N), chunksize=100
            )
        )
    pairs.sort(key=lambda p: p[0])
    return [p[1] for p in pairs]


start = time.time()
results = run_parallel_simulation(N)

# once we have the results, can we generate the data directly?
final_analysis_df = sampled_cohort[["place", "insurance"]].copy()
final_analysis_df["race"] = pd.Series(race_values)
final_analysis_df["sex"] = pd.Series(sex_values)
final_analysis_df["treatment_type"] = pd.Series(treatment_values)
final_analysis_df["years_to_death"] = pd.Series([r["years_to_death"] for r in results])
final_analysis_df["years_sick"] = pd.Series([r["years_sick"] for r in results])
final_analysis_df["years_treated"] = pd.Series([r["years_treated"] for r in results])
final_analysis_df["years_sick_treated"] = pd.Series(
    [r["years_sick_treated"] for r in results]
)
final_analysis_df["was_sick"] = pd.Series([r["was_sick"] for r in results])
final_analysis_df["was_treated"] = pd.Series([r["was_treated"] for r in results])
final_analysis_df["sick_age"] = pd.Series([r["sick_age"] for r in results])
final_analysis_df["treated_age"] = pd.Series([r["treated_age"] for r in results])

##stacked version of health system states
hs_traces = np.stack([r["hs_trace"] for r in results])
bp_traces = np.stack([r["bp_trace"] for r in results])

columns_trace = ["HS-Year " + str(x) for x in range(0, cycles + 1)]
hs_trace = pd.DataFrame(hs_traces, columns=columns_trace)

columns_trace2 = ["BPStageYear " + str(x) for x in range(0, cycles + 1)]
bp_trace = pd.DataFrame(bp_traces, columns=columns_trace2)
total_trace = pd.concat([hs_trace, bp_trace], axis=1)
total_trace["race"] = final_analysis_df["race"]
total_trace["insurance"] = final_analysis_df["insurance"]
total_trace["sex"] = final_analysis_df["sex"]

os.makedirs(f"{overall_folder}/results/{folder}/cohort_outcomes/social", exist_ok=True)
os.makedirs(f"{overall_folder}/figures/social", exist_ok=True)

final_analysis_df.to_csv(
    f"{overall_folder}/results/{folder}/cohort_outcomes/social/results_{assumption_path}.csv",
    index=False,
)

##run overall results
results_dict = create_overall_results(final_analysis_df)
with open(
    f"{overall_folder}/results/{folder}/cohort_outcomes/social/overall_{assumption_path}.pkl",
    "wb",
) as f:
    pickle.dump(results_dict, f)

results_race_dict = create_results_by_group(final_analysis_df, "race")
with open(
    f"{overall_folder}/results/{folder}/cohort_outcomes/social/race_group_{assumption_path}.pkl",
    "wb",
) as f:
    pickle.dump(results_race_dict, f)

results_race_insurance_dict = create_results_multiple_groups(
    final_analysis_df, "insurance"
)
with open(
    f"{overall_folder}/results/{folder}/cohort_outcomes/social/race_insurance_group_{assumption_path}.pkl",
    "wb",
) as f:
    pickle.dump(results_race_insurance_dict, f)

## for each scenario, we will create figures for:
# disease natural history prevalence,
# health system prevalence
# cumulative incidence of treatment by race/ethnicity
# cumulative incidence of hypertension control for more than one year

N_arr, E_arr, S1_arr, S2_arr, D_arr = plot_DNH_trace(
    total_trace,
    include_dead=True,
    folder=folder,
    analysis="social",
    save_path=f"{assumption_path}_with_dead",
)
N_arr, E_arr, S1_arr, S2_arr = plot_DNH_trace(
    total_trace,
    include_dead=False,
    folder=folder,
    analysis="social",
    save_path=f"{assumption_path}_without_dead",
)

plot_cum_treatment(
    total_trace, folder=folder, analysis="social", save_path=assumption_path
)

plot_cum_treatment_both_groups(
    total_trace, folder=folder, analysis="social", save_path=assumption_path
)

OHS_arr, IHS_arr, DT_arr, DUT_arr = run_HS_state_graph(
    total_trace, assumption_path, folder=folder, analysis="social", plot=True
)

plot_cum_incidence_control_hypertension(
    total_trace, folder, analysis="social", save_path=assumption_path
)

##age-specific prevalence of hypertension
##TO DO: we can also weight this by the census population for
# age-specific estimates
age_prev_hypertension = compute_weighted_htn_prevalence_by_age(
    total_trace,
    start_age=40,
    max_cycle=cycles,
    hs_prefix="HS-Year ",
    bp_prefix="BPStageYear ",
)

age_prev_hypertension.to_csv(
    f"{overall_folder}/results/{folder}/cohort_outcomes/social/hyp_prev_{assumption_path}.csv"
)

##overall incidence of stage 1 or stage 2 hypertension
htn_incidence = plot_cumulative_incidence_first_htn(
    total_trace,
    stage_prefix="BPStageYear ",
    sex_col="sex",
    race_col="race",
    max_cycle=cycles,
    start_age=40,
    female_value=1,
    race_values=("NHW", "NHB"),
)

htn_incidence.to_csv(
    f"{overall_folder}/results/{folder}/cohort_outcomes/social/hyp_inc_{assumption_path}.csv"
)

plot_hyp_incidence_by_group(htn_incidence, "social", assumption_path)

if assumption_path == "hetero_RA":
    plot_OHS_calibration_targets(total_trace, folder, assumption_path)

end = time.time()
print(end - start)

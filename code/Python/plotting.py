import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
import seaborn as sns

from numeric import CYCLE_LENGTH
from paths import overall_folder
from results_misc import cycles, starting_age
from stats import combine_se_errors


def plot_DNH_trace(total_trace, include_dead, folder, analysis, save_path):
    N_arr = []
    E_arr = []
    S1_arr = []
    S2_arr = []
    D_arr = []

    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]
    BP_state_trace_df_columns = ["BPStageYear " + str(x) for x in range(0, cycles + 1)]
    N = len(total_trace)

    if include_dead:
        for i in BP_state_trace_df_columns:
            N_arr.append(len(np.where(total_trace[i] == "N")[0]) / N)
            E_arr.append(len(np.where(total_trace[i] == "E")[0]) / N)
            S1_arr.append(len(np.where(total_trace[i] == "S1")[0]) / N)
            S2_arr.append(len(np.where(total_trace[i] == "S2")[0]) / N)
            D_arr.append(len(np.where(total_trace[i] == "D")[0]) / N)
        plt.figure(figsize=(8, 5))
        plt.plot(age_range, N_arr, label="Normal")
        plt.plot(age_range, E_arr, label="Elevated")
        plt.plot(age_range, S1_arr, label="Stage 1")
        plt.plot(age_range, S2_arr, label="Stage 2")
        plt.plot(age_range, D_arr, label="Dead")
        plt.legend()
        plt.ylabel("State proportion")
        plt.xlabel("Age")
        plt.savefig(
            f"{overall_folder}/figures/{analysis}/dnh_trace_{save_path}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        DNH_df = pd.DataFrame([N_arr, E_arr, S1_arr, S2_arr, D_arr]).T
        DNH_df.columns = ["N", "E", "S1", "S2", "D"]
        DNH_df.to_csv(
            f"{overall_folder}/results/{folder}/cohort_outcomes/{analysis}/DNH_prev_{save_path}.csv",
            index=False,
        )
        return N_arr, E_arr, S1_arr, S2_arr, D_arr
    else:
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

        plt.figure(figsize=(8, 5))
        plt.plot(age_range[: len(N_arr)], N_arr, label="Normal")
        plt.plot(age_range[: len(N_arr)], E_arr, label="Elevated")
        plt.plot(age_range[: len(N_arr)], S1_arr, label="Stage 1")
        plt.plot(age_range[: len(N_arr)], S2_arr, label="Stage 2")
        plt.legend()
        plt.ylabel("State prevalence")
        plt.xlabel("Age")
        plt.savefig(
            f"{overall_folder}/figures/{analysis}/dnh_trace_{save_path}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        DNH_df = pd.DataFrame([N_arr, E_arr, S1_arr, S2_arr]).T
        DNH_df.columns = ["N", "E", "S1", "S2"]
        DNH_df.to_csv(
            f"{overall_folder}/results/{folder}/cohort_outcomes/{analysis}/DNH_prev_{save_path}.csv",
            index=False,
        )
        return N_arr, E_arr, S1_arr, S2_arr


def run_HS_state_graph(trace, save_path, folder, analysis, plot=False):
    OHS_arr = []
    IHS_arr = []
    DT_arr = []
    DUT_arr = []

    cycle_cutoff = int(45 / CYCLE_LENGTH)
    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]
    for i in range(cycles + 1):
        trace_alive = trace[trace["BPStageYear " + str(i)] != "D"]
        alive_N = len(trace_alive)
        if alive_N > 0:
            OHS_arr.append(
                len(np.where(trace_alive["HS-Year " + str(i)] == "OHS")[0]) / alive_N
            )
            IHS_arr.append(
                len(np.where(trace_alive["HS-Year " + str(i)] == "IHS")[0]) / alive_N
            )
            DT_arr.append(
                len(np.where(trace_alive["HS-Year " + str(i)] == "DT")[0]) / alive_N
            )
            DUT_arr.append(
                len(np.where(trace_alive["HS-Year " + str(i)] == "DUT")[0]) / alive_N
            )
        else:
            OHS_arr.append(0)
            IHS_arr.append(0)
            DT_arr.append(0)
            DUT_arr.append(0)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(
            age_range[:cycle_cutoff],
            OHS_arr[:cycle_cutoff],
            color="red",
            label="Out of Health System",
        )
        plt.plot(
            age_range[:cycle_cutoff],
            IHS_arr[:cycle_cutoff],
            color="b",
            label="In Health System",
        )
        plt.plot(
            age_range[:cycle_cutoff],
            DT_arr[:cycle_cutoff],
            color="green",
            label="Detected/treated",
        )
        plt.plot(
            age_range[:cycle_cutoff],
            DUT_arr[:cycle_cutoff],
            color="orange",
            label="Detected/untreated",
        )
        plt.legend()
        plt.ylabel("State proportion")
        plt.xlabel("Age")
        plt.savefig(
            f"{overall_folder}/figures/{analysis}/health_system_trace_{save_path}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    HS_df = pd.DataFrame([OHS_arr, IHS_arr, DT_arr, DUT_arr]).T
    HS_df.columns = ["OHS", "IHS", "DT", "DUT"]
    HS_df.to_csv(
        f"{overall_folder}/results/{folder}/cohort_outcomes/{analysis}/HS_prev_{save_path}.csv",
        index=False,
    )
    return OHS_arr, IHS_arr, DT_arr, DUT_arr


def plot_cum_treatment(trace, folder, analysis, save_path):
    DT_arr = []

    cycle_cutoff = int(45 / CYCLE_LENGTH)
    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]
    columns_trace = ["HS-Year " + str(x) for x in range(0, cycles + 1)]

    for g in range(2):
        if g == 0:
            this_trace = trace[(trace["insurance"] == 1)]
        else:
            this_trace = trace[(trace["insurance"] == 0)]

        num_group = len(this_trace)
        hs_state = this_trace[columns_trace]
        hs_state = np.array(hs_state)

        first_index = [0 for i in range(cycles + 1)]
        for i in range(len(hs_state)):
            treat_idx = np.where(hs_state[i] == "DT")[0]
            if len(treat_idx) > 0:
                first_index[treat_idx[0]] = first_index[treat_idx[0]] + 1

        first_index = np.cumsum(first_index)
        this_arr = first_index / num_group
        DT_arr.append(this_arr)

    plt.figure(figsize=(8, 5))
    plt.plot(
        age_range[:cycle_cutoff],
        DT_arr[0][:cycle_cutoff],
        color="green",
        label="Insured",
    )
    if analysis == "social":
        plt.plot(
            age_range[:cycle_cutoff],
            DT_arr[1][:cycle_cutoff],
            color="green",
            ls="--",
            label="Uninsured",
        )
    plt.ylabel("Cumulative incidence of treatment")
    plt.xlabel("Age")

    bottom, top = plt.ylim()
    plt.ylim((bottom, 0.85))

    plt.legend()
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/cum_inc_treated_{save_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    DT_df = pd.DataFrame(DT_arr).T
    DT_df.columns = ["Insured", "Uninsured"]
    DT_df.to_csv(
        f"{overall_folder}/results/{folder}/cohort_outcomes/{analysis}/cum_inc_{save_path}.csv",
        index=False,
    )


def plot_cum_treatment_group(trace, group, folder, analysis, save_path):
    DT_arr = []

    cycle_cutoff = int(45 / CYCLE_LENGTH)
    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]
    columns_trace = ["HS-Year " + str(x) for x in range(0, cycles + 1)]

    for g in range(2):
        if g == 0:
            this_trace = trace[(trace["race"] == group) & (trace["insurance"] == 1)]
        else:
            this_trace = trace[(trace["race"] == group) & (trace["insurance"] == 0)]

        num_group = len(this_trace)
        hs_state = this_trace[columns_trace]
        hs_state = np.array(hs_state)

        first_index = [0 for i in range(cycles + 1)]
        for i in range(len(hs_state)):
            treat_idx = np.where(hs_state[i] == "DT")[0]
            if len(treat_idx) > 0:
                first_index[treat_idx[0]] = first_index[treat_idx[0]] + 1

        first_index = np.cumsum(first_index)
        this_arr = first_index / num_group
        DT_arr.append(this_arr)

    plt.figure(figsize=(8, 5))
    plt.plot(
        age_range[:cycle_cutoff],
        DT_arr[0][:cycle_cutoff],
        color="green",
        label="Insured",
    )
    if analysis == "social":
        plt.plot(
            age_range[:cycle_cutoff],
            DT_arr[1][:cycle_cutoff],
            color="green",
            ls="--",
            label="Uninsured",
        )
    plt.ylabel("Cumulative incidence of treatment")
    plt.xlabel("Age")

    bottom, top = plt.ylim()
    plt.ylim((bottom, 0.85))

    plt.legend()
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/cum_inc_treated_{save_path}_{group}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    DT_df = pd.DataFrame(DT_arr).T
    DT_df.columns = ["Insured", "Uninsured"]
    DT_df.to_csv(
        f"{overall_folder}/results/{folder}/cohort_outcomes/{analysis}/cum_inc_{save_path}_{group}.csv",
        index=False,
    )

    return DT_arr


def plot_cum_treatment_both_groups(trace, folder, analysis, save_path):
    race_order = ["Non-Hispanic Black", "Non-Hispanic white"]
    race_colors = dict(
        zip(
            race_order,
            sns.color_palette("Blues", n_colors=len(race_order)),
        )
    )
    nhb_color = race_colors["Non-Hispanic Black"]
    nhw_color = race_colors["Non-Hispanic white"]

    DT_arr_NHB = plot_cum_treatment_group(trace, "NHB", folder, analysis, save_path)
    DT_arr_NHW = plot_cum_treatment_group(trace, "NHW", folder, analysis, save_path)

    cycle_cutoff = int(45 / CYCLE_LENGTH)
    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]

    plt.figure(figsize=(8, 5))
    plt.plot(
        age_range[:cycle_cutoff],
        DT_arr_NHB[0][:cycle_cutoff],
        color=nhb_color,
        label="Non-Hispanic Black (Insured)",
    )
    plt.plot(
        age_range[:cycle_cutoff],
        DT_arr_NHW[0][:cycle_cutoff],
        color=nhw_color,
        label="Non-Hispanic white (Insured)",
    )
    if analysis == "social":
        plt.plot(
            age_range[:cycle_cutoff],
            DT_arr_NHB[1][:cycle_cutoff],
            color=nhb_color,
            ls="--",
            label="Non-Hispanic Black (Uninsured)",
        )
        plt.plot(
            age_range[:cycle_cutoff],
            DT_arr_NHW[1][:cycle_cutoff],
            color=nhw_color,
            ls="--",
            label="Non-Hispanic white(Uninsured)",
        )
    plt.ylabel("Cumulative incidence of treatment")
    plt.xlabel("Age")

    bottom, top = plt.ylim()
    plt.ylim((bottom, 0.85))

    plt.legend()
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/cum_inc_treated_{save_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def construct_OHS_calibration_targets():
    targets = pd.read_csv(
        f"{overall_folder}/data_and_models/nhanes_inputs/copula_outputs/nhanes_place_calibration_targets.csv"
    )
    age = [40, 50, 60, 70, 80]
    groups = targets["agegrp"].unique()[2:]

    targets_arr = []
    for idx, a in enumerate(groups):
        for i in [0, 1]:
            row_black = targets[
                (targets["agegrp"] == a)
                & (targets["insurance"] == i)
                & (targets["black"] == 1)
            ]
            row_non_black = targets[
                (targets["agegrp"] == a)
                & (targets["insurance"] == i)
                & (targets["black"] == 0)
            ]
            this_target = (
                0.5 * row_black["noplace"].values[0]
                + 0.5 * row_non_black["noplace"].values[0]
            )
            first_se = row_black["se"].values[0]
            second_se = row_non_black["se"].values[0]
            combined_se = combine_se_errors(first_se, second_se)

            targets_arr.append(
                [
                    a,
                    age[idx],
                    i,
                    this_target,
                    combined_se,
                    this_target - 1.96 * combined_se,
                    this_target + 1.96 * combined_se,
                ]
            )

    targets_arr = pd.DataFrame(
        targets_arr,
        columns=["age", "cycle", "insurance", "target", "se", "lower", "upper"],
    )
    return targets_arr


def plot_OHS_calibration_targets(trace, folder, save_path):
    targets_arr = construct_OHS_calibration_targets()

    OHS_arr_ins = []
    OHS_arr_no_ins = []
    trace_ins = trace[trace["insurance"] == 1]
    trace_no_ins = trace[trace["insurance"] == 0]

    cycle_cutoff = int(45 / CYCLE_LENGTH)
    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]

    for i in range(cycle_cutoff):
        trace_ins_alive = trace_ins[trace_ins["BPStageYear " + str(i)] != "D"]
        if len(trace_ins_alive) != 0:
            OHS_arr_ins.append(
                len(np.where(trace_ins_alive["HS-Year " + str(i)] == "OHS")[0])
                / len(trace_ins_alive)
            )
        trace_no_ins_alive = trace_no_ins[trace_no_ins["BPStageYear " + str(i)] != "D"]
        if len(trace_no_ins_alive) != 0:
            OHS_arr_no_ins.append(
                len(np.where(trace_no_ins_alive["HS-Year " + str(i)] == "OHS")[0])
                / len(trace_no_ins_alive)
            )

    plt.figure(figsize=(8, 5))
    plt.plot(
        age_range[: len(OHS_arr_ins)],
        OHS_arr_ins,
        color="green",
        label="With insurance",
    )
    plt.plot(
        age_range[: len(OHS_arr_no_ins)],
        OHS_arr_no_ins,
        color="red",
        label="Without insurance",
    )

    plt.errorbar(
        targets_arr[targets_arr["insurance"] == 1]["cycle"],
        targets_arr[targets_arr["insurance"] == 1]["target"],
        yerr=targets_arr[targets_arr["insurance"] == 1]["se"] * 1.96,
        color="green",
        alpha=0.5,
        linestyle="None",
    )

    plt.scatter(
        targets_arr[targets_arr["insurance"] == 1]["cycle"],
        targets_arr[targets_arr["insurance"] == 1]["target"],
        color="green",
        alpha=0.5,
    )
    plt.errorbar(
        targets_arr[targets_arr["insurance"] == 0]["cycle"].iloc[:3],
        targets_arr[targets_arr["insurance"] == 0]["target"].iloc[:3],
        yerr=targets_arr[targets_arr["insurance"] == 0]["se"].iloc[:3] * 1.96,
        color="red",
        alpha=0.5,
        linestyle="None",
    )

    plt.scatter(
        targets_arr[targets_arr["insurance"] == 0]["cycle"].iloc[:3],
        targets_arr[targets_arr["insurance"] == 0]["target"].iloc[:3],
        color="red",
        alpha=0.5,
    )
    plt.legend()
    plt.ylabel("Prevalence")
    plt.xlabel("Age")
    plt.savefig(
        f"{overall_folder}/figures/social/in_health_system_calibration_{save_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    OHS_df = pd.DataFrame([OHS_arr_ins, OHS_arr_no_ins]).T
    OHS_df.columns = ["Insured", "Uninsured"]
    OHS_df.to_csv(
        f"{overall_folder}/results/{folder}/cohort_outcomes/social/ohs_prev_{save_path}.csv",
        index=False,
    )


def plot_cumulative_treatment_exposure(
    kmf, DT_arr, DUT_arr, D_arr, N, analysis, save_path
):
    any_treated = [x + y for x, y in zip(DT_arr, DUT_arr)]
    cutoff = 36 * 12

    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]
    D_count = np.array(D_arr) * N

    plt.figure(figsize=(9, 6))
    plt.plot(range(40, 40 + len(kmf)), kmf, color="b", label="PRIME registry")
    plt.plot(
        age_range[:cutoff],
        any_treated[:cutoff],
        label="Simulation model",
        color="orange",
    )
    estimates_lower = [0 for i in range(cutoff)]
    estimates_upper = [0 for i in range(cutoff)]
    for i in range(cutoff):
        estimates_lower[i], estimates_upper[i] = (
            statsmodels.stats.proportion.proportion_confint(
                count=any_treated[i] * (N - D_count[i]),
                nobs=N - D_count[i],
                alpha=0.05,
                method="normal",
            )
        )
    plt.fill_between(
        age_range[:cutoff], estimates_lower, estimates_upper, color="orange", alpha=0.1
    )

    plt.legend()
    plt.xlabel("Age")
    plt.ylabel("Cumulative incidence (%)")
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/cum_treatment_exposure_{save_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_cum_incidence_control_hypertension(trace, folder, analysis, save_path):
    cycle_cutoff = int(45 / CYCLE_LENGTH)
    x = list(range(0, cycles + 1))
    age_range = [starting_age + i / 12 for i in x]
    columns_trace = ["HS-Year " + str(x) for x in range(0, cycles + 1)]
    columns_trace2 = ["BPStageYear " + str(x) for x in range(0, cycles + 1)]

    control_arr = []

    for g in range(4):
        if g == 0:
            this_trace = trace[(trace["race"] == "NHB") & (trace["insurance"] == 1)]
        elif g == 1:
            this_trace = trace[(trace["race"] == "NHB") & (trace["insurance"] == 0)]
        elif g == 2:
            this_trace = trace[(trace["race"] == "NHW") & (trace["insurance"] == 1)]
        else:
            this_trace = trace[(trace["race"] == "NHW") & (trace["insurance"] == 0)]

        hs_state = this_trace[columns_trace]
        hs_state = np.array(hs_state)
        bp_state = this_trace[columns_trace2]
        bp_state = np.array(bp_state)

        first_index_control = [0 for i in range(cycles + 1)]
        first_index = [0 for i in range(cycles + 1)]

        for i in range(len(hs_state)):
            treat_idx = np.where(hs_state[i] == "DT")[0]
            if len(treat_idx) > 0:
                first_index[treat_idx[0]] = first_index[treat_idx[0]] + 1
            no_hyp_idx = np.where((bp_state[i] == "N") | (bp_state[i] == "E"))[0]
            overlap = np.intersect1d(treat_idx, no_hyp_idx)
            if len(overlap) > 0:
                count = 1
                max_count = 1
                for j in range(1, len(overlap)):
                    if overlap[j] == overlap[j - 1] + 1:
                        count += 1
                    else:
                        max_count = max(max_count, count)
                        count = 1
                max_count = max(max_count, count)
                if max_count > 12:
                    first_index_control[overlap[0]] = (
                        first_index_control[overlap[0]] + 1
                    )

        first_index_control = np.cumsum(first_index_control)
        num_treated = sum(first_index)
        this_arr = first_index_control / num_treated

        control_arr.append(this_arr)

    control_df = pd.DataFrame(control_arr).T
    control_df.columns = [
        "NHB (insured)",
        "NHB (uninsured)",
        "NHW (insured)",
        "NHW (uninsured)",
    ]

    plt.figure(figsize=(10, 7))
    plt.plot(
        age_range[:cycle_cutoff],
        control_arr[0][:cycle_cutoff],
        color="blue",
        label="Non-Hispanic Black (insured)",
    )
    if analysis == "social":
        plt.plot(
            age_range[:cycle_cutoff],
            control_arr[1][:cycle_cutoff],
            color="blue",
            ls="--",
            label="Non-Hispanic Black (uninsured)",
        )
    plt.plot(
        age_range[:cycle_cutoff],
        control_arr[2][:cycle_cutoff],
        color="orange",
        label="Non-Hispanic white (insured)",
    )
    if analysis == "social":
        plt.plot(
            age_range[:cycle_cutoff],
            control_arr[3][:cycle_cutoff],
            color="orange",
            ls="--",
            label="Non-Hispanic white (uninsured)",
        )
    plt.ylabel("Cumulative incidence of hypertension control")
    plt.xlabel("Age")

    plt.legend(fontsize=14)
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/cum_inc_control_{save_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    control_df.to_csv(
        f"{overall_folder}/results/{folder}/cohort_outcomes/{analysis}/control_hyp_prev_{save_path}.csv",
        index=False,
    )


def plot_cumulative_incidence_first_htn(
    trace_df,
    stage_prefix="BPStageYear ",
    sex_col="sex",
    race_col="race",
    max_cycle=725,
    start_age=40,
    female_value=1,
    race_values=("NHW", "NHB"),
):
    df = trace_df.copy()

    n_individuals = len(df)

    first_event_cycle = np.full(n_individuals, np.nan)

    for c in range(max_cycle + 1):
        col = f"{stage_prefix}{c}"

        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

        state = df[col]

        alive = state != "D"
        event_now = alive & state.isin(["S1", "S2"])

        to_assign = np.isnan(first_event_cycle) & event_now.to_numpy()
        first_event_cycle[to_assign] = c

    df["first_event_cycle"] = first_event_cycle

    df["group"] = np.where(
        df[sex_col] == female_value, df[race_col] + " female", df[race_col] + " male"
    )

    valid_groups = [f"{r} female" for r in race_values] + [
        f"{r} male" for r in race_values
    ]
    df = df[df["group"].isin(valid_groups)].copy()

    rows = []

    for group, gdf in df.groupby("group"):
        n = len(gdf)
        if n == 0:
            continue

        first_cycles = gdf["first_event_cycle"].to_numpy()

        for c in range(max_cycle + 1):
            age = start_age + c / 12.0

            cum_inc = np.mean((~np.isnan(first_cycles)) & (first_cycles <= c))

            rows.append(
                {
                    "cycle": c,
                    "age": age,
                    "group": group,
                    "cumulative_incidence": cum_inc,
                }
            )

    out_df = pd.DataFrame(rows)

    out_df = out_df.set_index(["group", "age"]).sort_index()

    plt.figure(figsize=(9, 6))

    group_order = ["NHW female", "NHW male", "NHB female", "NHB male"]

    for group in group_order:
        if group in out_df.index.get_level_values(0):
            sub = out_df.loc[group].reset_index()

            plt.plot(sub["age"], sub["cumulative_incidence"], label=group)

    plt.xlabel("Age")
    plt.ylabel("Cumulative incidence of first S1/S2")
    plt.ylim(0, 1)
    plt.xlim(start_age, start_age + max_cycle / 12)
    plt.legend(title="")
    plt.tight_layout()
    plt.show()

    return out_df


def plot_hyp_incidence_by_group(hyp_inc, analysis, save_path):
    out_df = hyp_inc
    plt.figure(figsize=(9, 6))

    color_map = {
        "NHW female": "#1f77b4",
        "NHW male": "#ff7f0e",
        "NHB female": "#2ca02c",
        "NHB male": "#d62728",
    }

    group_order = ["NHB female", "NHB male", "NHW female", "NHW male"]
    group_description = {
        "NHB male": "Non-Hispanic Black men",
        "NHB female": "Non-Hispanic Black women",
        "NHW female": "Non-Hispanic white women",
        "NHW male": "Non-Hispanic white men",
    }

    for group in group_order:
        if group in out_df.index.get_level_values(0):
            sub = out_df.loc[group].reset_index()

            plt.plot(
                sub["age"].iloc[1:],
                sub["cumulative_incidence"].iloc[1:],
                color=color_map[group],
                label=group_description[group],
            )
    # taken from https://jamanetwork.com/journals/jamacardiology/fullarticle/2728380
    ages_knots = np.array([40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    white_women = np.array([27, 33, 40, 48, 56, 62, 66, 69, 69, 69])
    white_men = np.array([56, 62, 67, 73, 78, 81, 83, 84, 84, 84])
    black_women = np.array([42, 51, 63, 71, 77, 81, 84, 85, 85, 85])
    black_men = np.array([58, 64, 70, 76, 80, 83, 85, 86, 86, 86])

    white_women_p = white_women / 100
    white_men_p = white_men / 100
    black_women_p = black_women / 100
    black_men_p = black_men / 100

    err_white_women = np.full_like(white_women_p, 0.02)
    err_white_men = np.full_like(white_men_p, 0.02)
    err_black_women = np.full_like(black_women_p, 0.02)
    err_black_men = np.full_like(black_men_p, 0.02)

    plt.errorbar(
        ages_knots,
        white_women_p,
        yerr=err_white_women,
        fmt="o",
        capsize=3,
        color=color_map["NHW female"],
    )
    plt.errorbar(
        ages_knots,
        white_men_p,
        yerr=err_white_men,
        fmt="o",
        capsize=3,
        color=color_map["NHW male"],
    )
    plt.errorbar(
        ages_knots,
        black_women_p,
        yerr=err_black_women,
        fmt="o",
        capsize=3,
        color=color_map["NHB female"],
    )
    plt.errorbar(
        ages_knots,
        black_men_p,
        yerr=err_black_men,
        fmt="o",
        capsize=3,
        color=color_map["NHB male"],
    )

    plt.xlabel("Age")
    plt.ylabel("Cumulative incidence of first S1/S2")
    plt.ylim(0, 1)
    plt.xlim(starting_age, starting_age + cycles / 12)
    plt.legend(title="")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/hyp_inc_s1_s2_{save_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_treatment_incidence_from_age(
    final_analysis_df,
    start_age=55,
    end_age=85,
    female_value=1,
):
    df = final_analysis_df.copy()

    df.loc[df["was_treated"] == 0, "treated_age"] = np.nan

    eligible = (df["death_age"] > start_age) & (
        (df["treated_age"].isna()) | (df["treated_age"] > start_age)
    )
    df = df.loc[eligible].copy()
    df["group"] = np.where(df["sex"] == female_value, "Female", "Male")
    ages = np.arange(start_age, end_age + CYCLE_LENGTH, CYCLE_LENGTH)

    rows = []

    for group, sub in df.groupby("group"):
        n = len(sub)

        treated_age = sub["treated_age"].to_numpy(dtype=float)
        treated_age = np.where(np.isnan(treated_age), np.inf, treated_age)

        for age in ages:
            ci = np.mean((treated_age <= age) & (treated_age > start_age))
            rows.append(
                {
                    "group": group,
                    "age": age,
                    "years_since_55": age - start_age,
                    "cumulative_incidence": ci,
                    "n": n,
                }
            )

    return pd.DataFrame(rows)


def plot_treatment_validation_plot_by_sex(final_analysis_df, analysis, save_path):
    model_ci = create_treatment_incidence_from_age(final_analysis_df)

    paper_points = pd.DataFrame(
        {
            "group": (["Female"] * 4 + ["Male"] * 4),
            "years_since_55": [10, 15, 20, 25] * 2,
            "age": [65, 70, 75, 80] * 2,
            # https://jamanetwork.com/journals/jama/fullarticle/194679
            "estimate": [
                0.20,
                0.36,
                0.47,
                0.58,
                0.13,
                0.33,
                0.47,
                0.58,
            ],
            "lower": [
                0.15,
                0.31,
                0.43,
                0.54,
                0.09,
                0.28,
                0.41,
                0.53,
            ],
            "upper": [
                0.25,
                0.41,
                0.52,
                0.63,
                0.17,
                0.39,
                0.52,
                0.63,
            ],
        }
    )

    paper_points["yerr_lower"] = paper_points["estimate"] - paper_points["lower"]
    paper_points["yerr_upper"] = paper_points["upper"] - paper_points["estimate"]

    color_map = {
        "Female": "#1f77b4",
        "Male": "#ff7f0e",
    }

    plt.figure(figsize=(8, 5))

    for group in ["Female", "Male"]:
        sub = model_ci[model_ci["group"] == group]
        plt.plot(
            sub["years_since_55"],
            sub["cumulative_incidence"],
            label=f"{group}",
            color=color_map[group],
            linewidth=2,
        )

        pts = paper_points[paper_points["group"] == group]
        plt.errorbar(
            pts["years_since_55"],
            pts["estimate"],
            yerr=np.vstack([pts["yerr_lower"], pts["yerr_upper"]]),
            fmt="o",
            capsize=4,
            color=color_map[group],
        )

    plt.xlabel("Years since age 55")
    plt.ylabel("Cumulative incidence of antihypertensive therapy")
    plt.ylim(0, 1)
    plt.xlim(0, 30)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/cum_inc_hyp_55yo_{save_path}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

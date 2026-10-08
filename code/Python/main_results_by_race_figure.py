"""Generate the four 'main results by race' bar plots for one analysis arm.

Reads `results/<FOLDER>/cohort_outcomes/<analysis>/results_<assumption>.csv`
for each of four guideline assumptions (hetero_RA, hetero_NR, nohetero_RA,
nohetero_NR) and writes one PNG per outcome (years_sick, years_to_death,
years_sick_treated, years_treated) to `figures/<analysis>/`. Each plot
compares non-Hispanic Black and non-Hispanic white means side-by-side across
the four assumptions.

The standard and social arms differ in the `years_sick` filter: the standard
arm restricts to individuals who ever entered stage 1/2 hypertension, the
social arm does not. Other outcomes share filtering logic (years_treated and
years_sick_treated are restricted to the treated population in both arms).
"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

from paths import overall_folder

ASSUMPTIONS = ["hetero_RA", "hetero_NR", "nohetero_RA", "nohetero_NR"]
ASSUMPTION_LABELS = {
    "hetero_RA": "Race-\nadjusted",
    "hetero_NR": "No race\nadjustment",
    "nohetero_RA": "Race-\nadjusted ",
    "nohetero_NR": "No race\nadjustment ",
}
RACE_LABELS = {"NHB": "Non-Hispanic Black", "NHW": "Non-Hispanic white"}
OUTCOMES = ["years_sick", "years_to_death", "years_sick_treated", "years_treated"]


def build_outcome_frame(total_traces, outcome, analysis):
    """Stack per-assumption traces for one outcome into the long frame the
    seaborn catplot expects, applying outcome-specific row filters.
    """
    array_list = []
    for a, trace in zip(ASSUMPTIONS, total_traces):
        if outcome in ("years_sick_treated", "years_treated"):
            this_trace = trace[trace["was_treated"] == 1].reset_index()
        elif outcome == "years_sick" and analysis == "standard":
            this_trace = trace[trace["was_sick"] == 1].reset_index()
        else:
            this_trace = trace
        this_df = pd.DataFrame(this_trace[[outcome, "race"]]).copy()
        this_df["assumption"] = a
        array_list.append(this_df)
    total_df = pd.concat(array_list)
    total_df["race"] = total_df["race"].replace(RACE_LABELS)
    total_df["assumption"] = total_df["assumption"].replace(ASSUMPTION_LABELS)
    total_df.columns = [outcome, "Racial/Ethnic Group", "assumption"]
    return total_df


def plot_catplot_outcome(total_df, outcome, analysis, ylim=None):
    import seaborn as sns

    g = sns.catplot(
        data=total_df,
        kind="bar",
        x="assumption",
        y=outcome,
        hue="Racial/Ethnic Group",
        hue_order=["Non-Hispanic Black", "Non-Hispanic white"],
        errorbar=("ci", 0),
        palette="Blues",
        alpha=0.9,
        height=10,
        width=0.5,
        aspect=1.5,
    )
    if ylim is None:
        # Auto-compute ylim from per-bar means with 20% padding below the
        # smallest bar and 30% padding above the tallest, leaving room for the
        # legend that sits at upper center inside the plot. Pass an explicit
        # tuple to override.
        means = total_df.groupby(["assumption", "Racial/Ethnic Group"])[outcome].mean()
        data_min, data_max = means.min(), means.max()
        span = max(data_max - data_min, 0.5)
        if outcome == "years_sick":
            ylim = (data_min - span * 0.20, data_max + span * 0.4)
        else:
            ylim = (data_min - span * 0.20, data_max + span * 0.3)
    g.set(ylim=ylim)
    g.set_axis_labels("", "Years")

    # Keep the legend (at upper center) only on the years_to_death panel;
    # remove it elsewhere so the bars aren't crowded.
    if outcome == "years_sick":
        sns.move_legend(g, "upper center")
    else:
        g.legend.remove()

    ax = g.ax

    group_labels = [
        "Heterogeneous\ntreatment effects",
        "No heterogeneous\ntreatment effects",
    ]
    tick_positions = ax.get_xticks()

    group_positions = [
        (tick_positions[0] + tick_positions[1]) / 2,
        (tick_positions[2] + tick_positions[3]) / 2,
    ]

    ymin, ymax = ax.get_ylim()
    y_pos = ymin - (ymax - ymin) * 0.15

    for label, xpos in zip(group_labels, group_positions):
        ax.text(
            xpos, y_pos, label, ha="center", va="top", fontsize=30, fontweight="bold"
        )

    plt.subplots_adjust(bottom=0.2)
    plt.savefig(
        f"{overall_folder}/figures/{analysis}/main_results_by_race_{outcome}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--analysis", choices=["standard", "social"], required=True)
    parser.add_argument("-f", dest="folder", default="final_cohort")
    args = parser.parse_args()

    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    overall_folder = os.path.dirname(parent_directory)

    total_traces = []
    for a in ASSUMPTIONS:
        path = (
            f"{overall_folder}/results/{args.folder}/"
            f"cohort_outcomes/{args.analysis}/results_{a}.csv"
        )
        total_traces.append(pd.read_csv(path))

    os.makedirs(f"{overall_folder}/figures/{args.analysis}", exist_ok=True)

    with plt.rc_context({"font.size": 30}):
        for outcome in OUTCOMES:
            total_df = build_outcome_frame(total_traces, outcome, args.analysis)
            plot_catplot_outcome(total_df, outcome, args.analysis)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

from mortality import load_all_life_tables
from numeric import CYCLE_LENGTH
from paths import overall_folder

starting_age = 40
cycles = int((100 - starting_age) / CYCLE_LENGTH) + 5

life_table_dict = load_all_life_tables(overall_folder)


def simulate_death_age(race, sex, age):
    """
    Samples a death age for an individual based on a life table.

    Args:
        race: race label string ("NHB", "NHW", "H", or "O").
        sex: "FEMALE" or "MALE".
        age: starting age for the survival draw.
    """

    if sex == "FEMALE":
        sex_ind = 1
    else:
        sex_ind = 0

    # Indexed access raises KeyError on unknown (race, sex_ind).
    lt = life_table_dict[(race, sex_ind)]

    lt = lt[lt["age"] >= age].copy()
    lt["px"] = 1 - lt["qx"]
    lt["Sx"] = lt["px"].cumprod()
    lt["Sx"] = lt["Sx"] / lt["Sx"].iloc[0]

    u = np.random.uniform()

    death_age = lt[lt["Sx"] <= u]["age"].min()

    if pd.isna(death_age):
        death_age = lt["age"].max() + 1

    return float(death_age)

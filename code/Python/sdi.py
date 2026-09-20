import numpy as np


def calculate_likelihood(individual, area_data):
    probs = area_data[[f"pct_{char}" for char in individual.keys()]].values
    values = np.array(list(individual.values()))
    likelihoods = np.prod(np.where(values == 1, probs, 1 - probs), axis=1)
    return likelihoods


# Predicts the likelihood of each individual in the NHANES cohort to be from
# each of the SDI values in the data.
def return_SDI(row, area_data, SDI_AFC_df, rng):
    individual = row[
        [
            "Poverty_LT100",
            "Education_LT12years",
            "HH_Renter_Occupied",
            "NonEmployed",
            "HH_Crowding",
            "Single_Parent_Fam",
        ]
    ].to_dict()
    likelihoods = calculate_likelihood(individual, area_data)
    priors = area_data["prev"].values
    posterior_numerator = likelihoods * priors
    posterior = posterior_numerator / posterior_numerator.sum()
    area_data = area_data.assign(Posterior_Probability=posterior)

    random_list = rng.choice(
        area_data["CENSUSTRACT_FIPS"], size=10, p=area_data["Posterior_Probability"]
    )
    mean_SDI = SDI_AFC_df[SDI_AFC_df["CT"].isin(random_list)]["SDI_CT"].mean()
    return mean_SDI

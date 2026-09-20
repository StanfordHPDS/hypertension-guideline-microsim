import numpy as np
import pandas as pd
import statsmodels.api as sm

from numeric import logistic

# Column order for the measurement-frequency QuantReg design matrices. Models
# are fit positionally (statsmodels stores params as a vector), so prediction
# sites must project through the same list to avoid silently swapping
# coefficients.
MEASUREMENT_FREQ_COLS_STANDARD = [
    "intercept",
    "female",
    "age",
    "last_systolic_bp",
    "last_diastolic_bp",
]
MEASUREMENT_FREQ_COLS_SOCIAL = [
    "intercept",
    "female",
    "SDI",
    "age",
    "last_systolic_bp",
    "last_diastolic_bp",
]


def get_framework_folder(social):
    return "social" if social else "standard"


def load_quantile_models(overall_folder, social, quantiles):
    framework = get_framework_folder(social)

    sb_quantile_models = [
        sm.load(
            f"{overall_folder}/data_and_models/afc_models/{framework}/starting_systolic_bp_regressions/SBP_quantile_{q}_model.pickle"
        )
        for q in quantiles
    ]
    db_quantile_models = [
        sm.load(
            f"{overall_folder}/data_and_models/afc_models/{framework}/starting_diastolic_bp_regressions/DBP_quantile_{q}_model.pickle"
        )
        for q in quantiles
    ]
    return sb_quantile_models, db_quantile_models


def load_measurement_models(overall_folder, social, quantiles):
    framework = get_framework_folder(social)
    measurement_frequency_models = [
        sm.load(
            f"{overall_folder}/data_and_models/afc_models/{framework}/measurement_models_pre_treatment/measure_freq_quantile_{q}_model.pickle"
        )
        for q in quantiles
    ]
    return measurement_frequency_models


def load_splines(overall_folder, social):
    framework = get_framework_folder(social)

    systolic_splines = pd.read_csv(
        f"{overall_folder}/data_and_models/afc_outputs/{framework}/systolic_splines.csv"
    )
    diastolic_splines = pd.read_csv(
        f"{overall_folder}/data_and_models/afc_outputs/{framework}/diastolic_splines.csv"
    )

    return systolic_splines, diastolic_splines


def generate_new_SBP_value_SDI(
    sb_quantile_models, systolic_splines, this_quantile_index, age, sex, SDI
):
    this_model = sb_quantile_models[this_quantile_index]
    this_age = systolic_splines[
        (systolic_splines["age"] == round(age, 3)) & (systolic_splines["female"] == sex)
    ][systolic_splines.columns[2:]]
    this_pred = [1, sex, SDI] + list(this_age.iloc[0])
    return this_model.predict(pd.DataFrame(this_pred).T)[0]


def generate_new_SBP_value(
    sb_quantile_models, systolic_splines, this_quantile_index, age, sex
):
    this_model = sb_quantile_models[this_quantile_index]
    this_age = systolic_splines[
        (systolic_splines["age"] == round(age, 3)) & (systolic_splines["female"] == sex)
    ][systolic_splines.columns[2:]]
    this_pred = [1, sex] + list(this_age.iloc[0])
    return this_model.predict(pd.DataFrame(this_pred).T)[0]


def generate_new_DBP_value_SDI(
    db_quantile_models, diastolic_splines, this_quantile_index, age, sex, SDI
):
    this_model = db_quantile_models[this_quantile_index]
    this_age = diastolic_splines[
        (diastolic_splines["age"] == round(age, 3))
        & (diastolic_splines["female"] == sex)
    ][diastolic_splines.columns[2:]]
    this_pred = [1, sex, SDI] + list(this_age.iloc[0])
    return this_model.predict(pd.DataFrame(this_pred).T)[0]


def generate_new_DBP_value(
    db_quantile_models, diastolic_splines, this_quantile_index, age, sex
):
    this_model = db_quantile_models[this_quantile_index]
    this_age = diastolic_splines[
        (diastolic_splines["age"] == round(age, 3))
        & (diastolic_splines["female"] == sex)
    ][diastolic_splines.columns[2:]]
    this_pred = [1, sex] + list(this_age.iloc[0])
    return this_model.predict(pd.DataFrame(this_pred).T)[0]


def generate_hypertension_stage(new_SBP, new_DBP):
    new_SBP = round(new_SBP)
    new_DBP = round(new_DBP)
    if new_SBP >= 140 or new_DBP >= 90:
        return "S2"
    elif new_SBP >= 130 or new_DBP >= 80:
        return "S1"
    elif new_SBP >= 120 and new_DBP < 80:
        return "E"
    else:
        return "N"


# Predicts the number of months until the next blood pressure measurement.
def generate_measurement_frequency_SDI(
    measurement_frequency_models,
    this_quantile_index,
    age,
    sex,
    SDI,
    systolic_bp,
    diastolic_bp,
):
    this_model = measurement_frequency_models[this_quantile_index]
    this_pred = pd.DataFrame(
        [
            {
                "intercept": 1,
                "female": sex,
                "SDI": SDI,
                "age": age,
                "last_systolic_bp": systolic_bp,
                "last_diastolic_bp": diastolic_bp,
            }
        ]
    )[MEASUREMENT_FREQ_COLS_SOCIAL]
    return this_model.predict(this_pred)[0]


def generate_measurement_frequency(
    measurement_frequency_models,
    this_quantile_index,
    age,
    sex,
    systolic_bp,
    diastolic_bp,
):
    this_model = measurement_frequency_models[this_quantile_index]
    this_pred = pd.DataFrame(
        [
            {
                "intercept": 1,
                "female": sex,
                "age": age,
                "last_systolic_bp": systolic_bp,
                "last_diastolic_bp": diastolic_bp,
            }
        ]
    )[MEASUREMENT_FREQ_COLS_STANDARD]
    return this_model.predict(this_pred)[0]


# Predicts the probability of initiating treatment at a measurement visit.
def generate_treatment_prob(
    logit_model,
    age,
    sex,
    systolic_bp,
    diastolic_bp,
):
    this_pred = [1, age, sex, systolic_bp, diastolic_bp]
    linear_combination = np.dot(this_pred, logit_model)
    return logistic(linear_combination)


def generate_treatment_prob_SDI(
    logit_model,
    age,
    sex,
    SDI,
    systolic_bp,
    diastolic_bp,
):
    this_pred = [1, age, sex, SDI, systolic_bp, diastolic_bp]
    linear_combination = np.dot(this_pred, logit_model)
    return logistic(linear_combination)

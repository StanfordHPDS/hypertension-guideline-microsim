import numpy as np

# Monthly cycle length, expressed as a fraction of a year.
CYCLE_LENGTH = 1 / 12.0


def convert_to_rate(prob):
    return -np.log(1 - prob)


def convert_to_prob(rate):
    return 1 - np.exp(-rate)


def convert_to_cycle(prob):
    annual_rate = -np.log(1.0 - prob)
    monthly_rate = annual_rate * CYCLE_LENGTH
    return 1 - np.exp(-monthly_rate)


def convert_to_percent(value):
    return int(round((value * 100)))


def logistic(z):
    return 1 / (1 + np.exp(-z))


def round_to_nearest_0_01(number):
    return round(number * 100) / 100

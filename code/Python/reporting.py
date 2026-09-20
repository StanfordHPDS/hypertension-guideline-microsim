from numeric import convert_to_percent

ROUND_DIGITS = 1


def create_95_CI_mean_se(mean, se, unit, digits=ROUND_DIGITS):
    return f"{round(mean, digits)} {unit} (95% CI: {round(mean - 1.96 * se, digits)}, {round(mean + 1.96 * se, digits)})"


def create_95_CI(value, unit, digits=ROUND_DIGITS):
    return f"{round(value[0], digits)} {unit} (95% CI: {round(value[0] - 1.96 * value[1], digits)}, {round(value[0] + 1.96 * value[1], digits)})"


def create_95_CI_percents(value):
    return f"{convert_to_percent(value[0])}% (95% CI: {convert_to_percent(value[0] - 1.96 * value[1])}%, {convert_to_percent(value[0] + 1.96 * value[1])}%)"


def create_95_CI_percents_mean_se(mean, se):
    return f"{convert_to_percent(mean)}% (95% CI: {convert_to_percent(mean - 1.96 * se)}%, {convert_to_percent(mean + 1.96 * se)}%)"


def format_value_and_se(value, digits=ROUND_DIGITS, threshold=0.01):
    v = round(value[0], digits)
    s = round(value[1], digits)
    if threshold is not None and s < threshold:
        se_str = f"<{threshold}"
    else:
        se_str = f"{s}"
    return f"{v} [{se_str}]"


def format_percent(value, digits=0, threshold=1):
    rounded = convert_to_percent(value[0])
    s = convert_to_percent(value[1])
    if s < threshold:
        se_str = f"<{threshold}"
    else:
        se_str = f"{s}"
    return f"{rounded}% [{se_str}%]"


def format_p_value(p, digits=3):
    p = float(p)

    if p < 0.001:
        return "<0.001"
    else:
        return f"{p:.{digits}f}"

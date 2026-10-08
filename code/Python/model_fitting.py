from concurrent.futures import ThreadPoolExecutor, as_completed

import dask.dataframe as dd
import numpy as np
import pandas as pd


def predict_model(model, X):
    return model.predict(X)


def parallel_predict_models(models, X, n_jobs=None):
    results = [None] * len(models)
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(predict_model, model, X): idx
            for idx, model in enumerate(models)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return np.column_stack(results)


# taken from https://rdrr.io/cran/WRTDStidal/src/R/goodfit.R
def quant_goodfit(resid, resid_nl, tau):
    resid = np.asarray(resid)
    resid_nl = np.asarray(resid_nl)

    V1 = resid * (tau - (resid < 0))
    V1 = np.nansum(V1)

    V0 = resid_nl * (tau - (resid_nl < 0))
    V0 = np.nansum(V0)

    try:
        out = 1 - V1 / V0
    except ZeroDivisionError:
        return np.nan

    if np.isinf(out) or V1 > V0:
        out = np.nan

    return out


def build_predictor_df(df, spline_cols, social):
    if social:
        X = pd.DataFrame({"const": 1, "sex": df["female"], "SDI": df["SDI"]})
    else:
        X = pd.DataFrame({"const": 1, "sex": df["female"]})

    for col in spline_cols:
        X[col] = df[col]

    return X


def raise_if_nan(arr, name):
    n = int(np.isnan(arr).sum())
    if n:
        raise ValueError(
            f"{name}: {n} NaN value(s); cannot determine a quantile assignment."
        )


def load_and_concat_parquet(folder, file_end, prefix):
    dfs = [
        dd.read_parquet(f"results/{folder}/saved_results/{prefix}_{file_end}{suffix}/")
        for suffix in ["1", "2"]
    ]
    return dd.concat(dfs, axis=0)

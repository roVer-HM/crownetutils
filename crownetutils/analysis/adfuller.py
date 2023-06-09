import pandas as pd


def adf_test(data: pd.Series):
    from statsmodels.tsa.stattools import adfuller

    print("Dickey-Fuller Test:")
    if isinstance(data, pd.DataFrame):
        df_test = adfuller(data.iloc[:, 0].to_numpy(), autolag="AIC")
    else:
        df_test = adfuller(data.to_numpy(), autolag="AIC")
    out = pd.Series(
        df_test[0:4], index=["Test Statistic", "p-value", "Lags Used", "NumObsUsed"]
    )
    out["tMin"] = data.index.min()
    out["tMax"] = data.index.max()
    for k, v in df_test[4].items():
        out[f"Critical Value ({k})"] = v
    # print(out)
    return out


def adf_summary_test(data: pd.DataFrame, col):
    _adf = adf_test(data.loc[:, [col]])
    _adf.name = "adf"
    _adf = _adf.to_frame()
    _adf.columns = pd.Index([col])
    _stat = data.loc[:, [col]].describe()
    df = pd.concat([_adf, _stat], axis=0)
    return df

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage

plt.get_cmap("tab20")


def format_predictions(df:pd.DataFrame) -> pd.DataFrame:

    df_formated = df[["Client","Warehouse","Product","Date","y_pred"]].copy()
    df_formated.Date = df_formated.Date.dt.date

    df_pivoted = df_formated.pivot(columns="Date", index=["Client","Warehouse","Product"], values="y_pred").reset_index()
    # df_pivoted.drop("Date", axis=1, inplace=True)

    df_pivoted.fillna(0, inplace=True)

    df_pivoted.to_csv(r"..\data\raw\Submission Phase 1 - CDLB.csv", index=False)
    
    return df_pivoted


def plot_predictions(df_train:pd.DataFrame, df_test:pd.DataFrame, y_pred:np.ndarray, feature:str, training_purpose:bool=True):

    df_test["y_pred"] = y_pred

    df_train= df_train.groupby("Date").agg(sum_col=(feature, "sum")).rename(columns={"sum_col": feature})
    df_test = df_test.groupby("Date").agg(sum_col1=(feature, "sum"),
                                          sum_col2=("y_pred", "sum")).rename(columns={"sum_col1": feature,
                                                                                      "sum_col2":"y_pred"})

    plt.figure(figsize=(10, 6))


    if training_purpose:
        df_grouped = pd.concat([df_train,df_test], axis=0)
        plt.plot(df_grouped.index, df_grouped[feature], label="Données Réelles", color="C2", marker="o", linestyle="-")
    
    else:
        plt.plot(df_train.index, df_train[feature], label="Données Réelles", color="C2", marker="o", linestyle="-")


    plt.plot(df_test.index, df_test.y_pred, label="Prédictions", color="C1", marker="x", linestyle="--")
    plt.axvline(df_test.index.min(), color="black", ls="--")

    plt.xlabel("Date")
    plt.ylabel(feature)

    plt.grid(True)
    plt.show()


def plotSeasonalDecompose(
    x,
    model='additive',
    filt=None,
    period=None,
    two_sided=True,
    extrapolate_trend=0,
    title="Seasonal Decomposition"):


    result = seasonal_decompose(
            x, model=model, filt=filt, period=period,
            two_sided=two_sided, extrapolate_trend=extrapolate_trend)
    fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"])
    for idx, col in enumerate(['observed', 'trend', 'seasonal', 'resid']):
        fig.add_trace(
            go.Scatter(x=result.observed.index, y=getattr(result, col), mode='lines'),
                row=idx+1, col=1,
            )
    return fig



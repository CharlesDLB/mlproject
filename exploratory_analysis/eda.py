import logging
import datetime as dt
from skimpy import skim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsforecast.utils import ConformalIntervals
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA

from src.logger.logger import setup_logger
from src.features.add_features import add_features, add_statistical_ft, add_trend_ft, ft_subset, add_consecutive_weeks_without_sales
from src.visualisation.visualise import heatmap, plotSeasonalDecompose

# import warnings
# warnings.filterwarnings("ignore")

# #################################################################################################
# LOADING DATA
# #################################################################################################

logger = setup_logger(__name__, logging.INFO, stream=True)

logger.info("-------------------------")
logger.info("Loading data")

df_raw = pd.read_parquet(r"data\2_cleaned\data.parquet")

df_raw["Price"] = df_raw["Price"].replace(0, np.nan)
df_raw.Date = pd.to_datetime(df_raw.Date)
df_raw.sort_values(by=["Client","Warehouse","Product","Date"], inplace=True)

KEY_FT = ["Client","Warehouse","Product"]
TEST_DATE = dt.datetime(2023,10,2)
MAX_DATE = max(df_raw.Date)

df_raw["Is_future"] = np.where(df_raw.Date > TEST_DATE, 1,0)

df_raw["Key"] = df_raw["Client"].astype(str) + "_" + df_raw["Warehouse"].astype(str) + "_" + df_raw["Product"].astype(str)


keys_subsets = ft_subset(["Client","Warehouse","Product"])

df = add_features(df_raw, logger=logger)
df = add_statistical_ft(df, "Sales", keys_subsets, ["sum","cumsum"], timeframe=["Date"], logger=logger)
df = add_trend_ft(df, ["Client","Warehouse","Product"], logger=logger)


# #################################################################################################
# Plot Features correlations
# #################################################################################################


cols = [col for col in list(df.columns) 
        if col not in ["Game_phase","Is_future","Is_sales","Date","Key","Client","Warehouse","Product"]
        and "dist" not in col]

correlation_matrix = df[cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Corr matrix")
plt.show()


correlation_matrix = pd.melt(correlation_matrix.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
correlation_matrix.columns = ['x', 'y', 'value']
heatmap(
    x=correlation_matrix['x'],
    y=correlation_matrix['y'],
    size=correlation_matrix['value'].abs(),
)


# #################################################################################################
# Plot ACF, PACF graph
# #################################################################################################

df = df_raw.loc[df_raw.Key.isin(["41_206_8986","38_63_12903","41_228_2866"])].copy()


df_train = df.loc[df.ds <= dt.datetime(2023,10,2)]
df_test = df.loc[df.ds > dt.datetime(2023,10,2)]

fig, axs = plt.subplots(nrows=1, ncols=2)

plot_acf(df_train["y"],  lags=30, ax=axs[0],color="fuchsia")

plot_pacf(df_train["y"],  lags=30, ax=axs[1],color="lime")

axs[0].set_title("Autocorrelation")
axs[1].set_title('Partial Autocorrelation')

plt.show()

plotSeasonalDecompose(
    df_train["y"],
    model="additive",
    period=52,
    title="Seasonal Decomposition")

plotSeasonalDecompose(
    df_train["y"],
    model="multiplicative",
    period=52,
    title="Seasonal Decomposition")

horizon = len(df_test) # number of predictions

models = [MSTL(season_length=[52], # seasonalities of the time series
               trend_forecaster=AutoARIMA(
                   prediction_intervals=ConformalIntervals(n_windows=3, h=horizon)))
        ]

sf = StatsForecast(models=models, freq='W')

sf.fit(df=df_train)


result=sf.fitted_[0,0].model_

result

sf.fitted_[0, 0].model_.tail(24 * 28).plot(subplots=True, grid=True)
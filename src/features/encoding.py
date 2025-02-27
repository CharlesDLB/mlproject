import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import SplineTransformer

from typing import Any

def _periodic_spline_transformer(period:int, n_splines:int=None, degree:int=3) -> SplineTransformer:
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True
        )


def spline_encoding(df:pd.DataFrame, feature_name:str, period:int, n_splines:int=None) -> pd.DataFrame:
    feature_name = feature_name.capitalize()

    encoder = _periodic_spline_transformer(period, n_splines)
    encoder.fit(df[[feature_name]])
    values = encoder.transform(df[[feature_name]])

    columns = []
    for i in range(n_splines):
        columns.append(f"{feature_name}_spline{i+1}")

    df_spline = pd.DataFrame(values, columns=columns)

    df = pd.concat([df,df_spline], axis=1)

    return df


def trigo_encoding(df:pd.DataFrame, feature_list:list[str], period:int) -> pd.DataFrame:

    for ft in feature_list:
        df[ft + "_sin"] = round(np.sin(2 * np.pi * df[ft]/period),3)
        df[ft + "_cos"] = round(np.cos(2 * np.pi * df[ft]/period),3)

    return df


def ordinal_encoding(df:pd.DataFrame, ft_to_encode:list[str]) -> pd.DataFrame:

    encoder = OrdinalEncoder()

    df[ft_to_encode] = encoder.fit_transform(df[ft_to_encode])

    df[ft_to_encode] = df[ft_to_encode].astype(int)

    return df


def encode_features(ordinal_ft:list[str], trigo_ft:list[str], spline_ft:dict[str:tuple[int,int]], logger:logging.Logger) -> pd.DataFrame:
    """spline_ft example : {"Week":(period=52, nb_of_spline=13)}"""

    logger.info("-------------------------")
    logger.info("Feature encoding")
    logger.info("-> Ordinal")
    df = ordinal_encoding(df, ordinal_ft)

    logger.info("-> Tigonometric")
    df = trigo_encoding(df, trigo_ft, 12)

    logger.info("-> Spline")
    
    for feature in spline_ft:
        df = spline_encoding(df, feature_name=feature, period=spline_ft[feature][0], n_splines=spline_ft[feature][1])

    return df




def scale_features(df_train:pd.DataFrame, ft_to_scale:list[str], scaler:Any, df_test:pd.DataFrame=None, duplicate_cols:bool=True) -> pd.DataFrame:

    if duplicate_cols == True:
        new_cols = []
        for col in ft_to_scale:
            new_cols.append(col + "_scaled")
        df_train[new_cols] = scaler.fit_transform(df_train[ft_to_scale])
        if df_test is not None:
            df_test[new_cols] = scaler.transform(df_test[ft_to_scale])
    else:
        df_train[ft_to_scale] = scaler.fit_transform(df_train[ft_to_scale])
        if df_test is not None:
            df_test[ft_to_scale] = scaler.transform(df_test[ft_to_scale])

    return df_train, df_test


def shift(df:pd.DataFrame, grouping_features:list[str], columns_to_shift:list[str], shift_period:int, duplicate_cols:bool=False) -> pd.DataFrame:

    if duplicate_cols == True:
        for feature in columns_to_shift:
            df[f"{feature}_shifted"] = df.groupby(grouping_features)[feature].shift(shift_period)
    else:
        for feature in columns_to_shift:
            df[feature] = df.groupby(grouping_features)[feature].shift(shift_period)

    return df



if __name__ == "__main__":

    df = pd.read_parquet(r"..\..\data\processed\features_data.parquet")
    df.shape

    df = trigo_encoding(df, "Month", 12)

    df = ordinal_encoding(df, ["Year"])







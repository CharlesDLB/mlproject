import pandas as pd
import numpy as np

def price_imputation(df:pd.DataFrame, duplicate_col:bool = False) -> pd.DataFrame:
    """Require temporal features
    Warning : use for each cv split to avoid data leakage"""

    df["Price"] = df["Price"].replace(0, np.nan)
    
    df["Price_1"] = df.groupby(["Warehouse","Product","Year","Week"])["Price"].transform("mean")
    df["Price_2"] = df.groupby(["Product","Year","Week"])["Price"].transform("mean")
    df["Price_3"] = df.groupby(["Product","Year","Month"])["Price"].transform("mean")
    df["Price_4"] = df.groupby(["Product","Week"])["Price"].transform("mean")
    df["Price_5"] = df.groupby(["Product","Month"])["Price"].transform("mean")
    df["Price_6"] = df.groupby(["Product"])["Price"].transform("mean")
    df["Price_7"] = np.mean(df.Price)

    df["Price_imputed"] = (df.Price.fillna(df.Price_1)
                           .fillna(df.Price_2)
                           .fillna(df.Price_3)
                           .fillna(df.Price_4)
                           .fillna(df.Price_5)
                           .fillna(df.Price_6)
                           .fillna(df.Price_7))
    
    if duplicate_col is True:
        df.drop(["Price_1","Price_2","Price_3","Price_4","Price_5","Price_6","Price_7"], axis=1, inplace=True)
        return df
    
    df["Price"] = df["Price_imputed"]

    df.drop(["Price_imputed","Price_1","Price_2","Price_3","Price_4","Price_5","Price_6","Price_7"], axis=1, inplace=True)

    return df


def drop_starting_zeros(df:pd.DataFrame, filtering_feature:str) -> pd.DataFrame:
    """Delete every raws with sales=0 before first sale
    """
    df.sort_values(by=[filtering_feature,"Date"], inplace=True)
    df["Cumulative_sales"] = df.groupby(filtering_feature)["Sales"].cumsum()
    df = df[df["Cumulative_sales"]>0]
    df.drop("Cumulative_sales", axis=1, inplace=True)
    df.sort_values(by=["Client","Warehouse","Product","Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

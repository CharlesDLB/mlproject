import pandas as pd


def merge_sales_price(df:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:

    df = pd.merge(df, df2, on=["Client","Warehouse","Product","Date"], how="left")
    df = df.sort_values(by=["Client","Warehouse","Product","Date"])
    df.reset_index(drop=True, inplace=True)
    df.drop("Phase_y", axis=1, inplace=True)
    df.rename(columns={"Phase_x":"Game_phase"}, inplace=True)

    return df




if __name__ == "__main__":

    SALES = r"..\..\data\1_serialized\sales.parquet"
    PRICE = r"..\..\data\1_serialized\price.parquet"
   
    df_sales = pd.read_parquet(SALES)
    df_price = pd.read_parquet(PRICE)


    df = merge_sales_price(df_sales, df_price)
    df.to_parquet(r"..\..\data\2_cleaned\data.parquet")

import pandas as pd
import datetime as dt


def time_serie_split(df:pd.DataFrame, predict_week_range:int, splits:int=5) -> list[tuple]:
    """
    Returns: list[tuple]: [(train_idx, val_idx), (train_idx, val_idx), ... ]
    """
    
    splits_list = []
    date_max = df["Date"].max()

    for i in range(splits):
        left_bound = date_max - dt.timedelta(weeks= (splits-i) * predict_week_range)
        right_bound = date_max - dt.timedelta(weeks= (splits-i-1) * predict_week_range)

        train_idx = df.loc[(df.Date <= left_bound)].index
        val_idx = df.loc[(df.Date > left_bound) & (df.Date <= right_bound)].index

        splits_list.append((train_idx,val_idx))

    return splits_list




if __name__ == "__main__":

    df_init = pd.read_parquet(r"..\..\data\raw\raw_data.parquet")

    # Create Training and Test
    df = df_init.copy()
    # df.fillna(0, inplace=True)

    df.sort_values(by=["Date","Client","Warehouse","Product"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    ts_splits = time_serie_split(df, predict_week_range=13, splits=1)
    # plot_cv_splits(df, ts_splits)






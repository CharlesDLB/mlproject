import pandas as pd
import numpy as np
import datetime as dt
import itertools
import logging

# from ..data.get_data import get_weather_data

# Create subsets of features
def ft_subset(grouping_features:list) -> list[list]:
        """Create groupby combinations of 1 and 2 features
        """
        feature_subsets = []
        for i in grouping_features:
            feature_subsets.append([i])

        combinations = list(itertools.combinations(grouping_features, 2))
        for i in combinations:
            feature_subsets.append(list(i))
        
        return feature_subsets


# Temporal features
def _year_correction(df:pd.DataFrame) -> pd.DataFrame:
    for i in range(len(df)):
        
        if df.loc[i, 'Week'] == 1:
            if i + 1 < len(df):  
                df.loc[i, 'Year'] = df.loc[i + 1, 'Year']
        
        if df.loc[i, 'Week'] in [52, 53]:
            if i - 2 >= 0:
                df.loc[i, 'Year'] = df.loc[i - 2, 'Year']
    
    return df

def _get_month_from_week(year:int, week:int) -> int:
    
    monday = dt.datetime.fromisocalendar(year, week, 1)  # 1 = lundi

    thursday = dt.datetime.fromisocalendar(year, week, 4)  # 4 = jeudi

    if monday.month == thursday.month:
        return monday.month
    else:
        return thursday.month

def _month_correction(df:pd.DataFrame) -> pd.DataFrame:
    for i in range(len(df)):
        
        if df.loc[i, 'Week'] == 1:
            df.loc[i, 'Month'] = 1
        
        if df.loc[i, 'Week'] in [52, 53]:
            df.loc[i, 'Month'] = 12
    
    return df

def add_temporal_ft(df:pd.DataFrame) -> pd.DataFrame:
    df.Date = pd.to_datetime(df.Date)
    df_date = pd.DataFrame({"Date":df["Date"].unique()})
    df_date["Year"] = df_date["Date"].dt.year
    df_date["Week"] = df_date["Date"].dt.isocalendar().week
    df_date = _year_correction(df_date)
    df_date['Month'] = df_date.apply(lambda row: _get_month_from_week(row['Year'], row['Week']), axis=1)
    df_date = _month_correction(df_date)

    df = pd.merge(df, df_date, how="left", on="Date")

    return df


# Calendar features
def get_black_friday_and_cyber_monday_weeks() -> list[dt.date]:

    black_friday_list = []
    cyber_monday_list = []

    for year in range(2010,2050): 
        first_november_day = dt.date(year, 11, 1)
        last_november_day = dt.date(year, 11, 30)

        if first_november_day.weekday() == 3: #jeudi
            black_friday = last_november_day - dt.timedelta(weeks=1)

        else:
            while last_november_day.weekday() != 4: 
                last_november_day -= dt.timedelta(days=1)
            black_friday = last_november_day

        cyber_monday = black_friday + dt.timedelta(days=3)

        black_friday_list.append(black_friday)
        cyber_monday_list.append(cyber_monday)


    return black_friday_list, cyber_monday_list

def get_chinese_new_year_dates() -> list[dt.date]:
    new_year_dates = [
        '2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05',
        '2020-01-25', '2021-02-12', '2022-02-01', '2023-01-22', 
        '2024-02-10', '2025-01-29', '2026-02-17', '2027-02-06', 
        '2028-01-26', '2029-02-13', '2030-02-03', '2031-01-23'
        ]

    new_year_dates = pd.to_datetime(new_year_dates).date

    return new_year_dates

def _add_event_column(df:pd.DataFrame, event_name:str, month:int=None, day:int=None, list_of_dates:list[dt.date]=None) -> pd.DataFrame:

    if list_of_dates is not None :
        date_range = pd.to_datetime(list_of_dates)

    else:
        date_range = pd.to_datetime([dt.date(i, month, day) for i in range(2010,2030)])
    

    date_range = date_range - pd.to_timedelta(date_range.weekday, unit='d')
    df[event_name] = df['Date'].isin(date_range).astype(int).astype("category")

    return df

def _add_days_diff_column(df:pd.DataFrame, column_name:str) -> pd.DataFrame:

    df["ffill"] = df["Date"].where(df[column_name] == 1).ffill().fillna(dt.datetime(2017,1,1))

    df["bfill"] = df["Date"].where(df[column_name] == 1).bfill().fillna(dt.datetime(2017,1,1))

    df[f"{column_name}_dist"] = np.minimum((df["Date"] - df["ffill"]).abs(),
                                           (df["Date"] - df["bfill"]).abs()
                                           )
    
    df[f"{column_name}_dist"] = df[f"{column_name}_dist"].dt.days.astype(int)
    
    df.drop(["ffill","bfill"], axis=1, inplace=True)

    return df

fix_event_dict = {"christmas_day": (12,25),
                "new_year_day": (1,1),
                "valentines_day": (2,14),
                "boxing_day": (12,26),
                "chinese_sales": (11,11),
                "summer_sales": (7,1)
                }

black_friday_list, cyber_monday_list = get_black_friday_and_cyber_monday_weeks()
chinese_new_year_list = get_chinese_new_year_dates()

var_event_dict = {"chinese_new_year": chinese_new_year_list,
                "black_friday": black_friday_list,
                "cyber_monday": cyber_monday_list
                }

def add_calendar_ft(df:pd.DataFrame, fix_event_dict:dict=None, var_event_dict:dict=None, add_distance_cols:bool=False) -> pd.DataFrame:
    """Add binary categorical columns for fixed event like Christmas and variable event like chinese new year,
    Can also add distance in days from nearest occurence

    Args:
        df (pd.DataFrame): original database
        fix_event_dict (dict, optional): {event_name:(month, day)} -> dict for fix day event. Defaults to None.
        var_event_dict (dict, optional): {event_name:list of dates} -> dict for var event. Defaults to None.

    Returns:
        pd.DataFrame: augmented dataframe with new features
    """
    date_range = pd.date_range(dt.datetime(2017,1,1), dt.datetime(2027,12,31), freq="W-MON")
    df_date = pd.DataFrame({"Date":date_range})

    if fix_event_dict is not None:
        for event_name in fix_event_dict:
            df_date = _add_event_column(df_date, event_name=event_name, month=fix_event_dict[event_name][0], day=fix_event_dict[event_name][1])
            if add_distance_cols == True:
                df_date = _add_days_diff_column(df_date, event_name)
                df_date.drop(event_name, axis=1, inplace=True)
    
    if var_event_dict is not None:
        for event_name in var_event_dict:
            df_date = _add_event_column(df_date, event_name=event_name, list_of_dates=var_event_dict[event_name])
            if add_distance_cols == True:
                df_date = _add_days_diff_column(df_date, event_name)
                df_date.drop(event_name, axis=1, inplace=True)
        
    
    df = pd.merge(df,df_date, on="Date", how="left")

    return df

def add_features(df:pd.DataFrame, logger:logging.Logger) -> pd.DataFrame:
    """Add Revenue, Is_sales, Temporal ft, Weather ft and Calendar ft"""

    logger.info("-> Revenue")
    df["Revenue"] = df["Sales"] * df["Price"]

    logger.info("-> Is_sales")
    df["Is_sales"] = np.where(df["Sales"]>0 ,1,0)

    logger.info("-> Temporal")
    df = add_temporal_ft(df)

    # logger.info("-> Weather")
    # df_weather = get_weather_data("2019-12-20", "2024-06-15")
    # df = pd.merge(df, df_weather, on="Date", how="left")

    logger.info("-> Calendar")
    df = add_calendar_ft(df, fix_event_dict=fix_event_dict, var_event_dict=var_event_dict, add_distance_cols=True)

    return df

def avg_consecutive_weeks_without_sales(sales_series:pd.Series) -> float:
    
    sales = sales_series.values
    
    if sum(sales) == 0:
        return 0.0
    
    start_index = np.argmax(sales > 0)
    sales = sales[start_index:]

    weeks_without_sales = []
    count = 0
    for sale in sales:
        if sale == 0:
            count += 1
        else:
            if count > 0:
                weeks_without_sales.append(count)
            count = 0
    if count > 0:
        weeks_without_sales.append(count)

    if not weeks_without_sales:
        return 0.0

    return np.mean(weeks_without_sales)

# Statistical features
def add_statistical_ft(df:pd.DataFrame, feature:str, groupby_subsets:list[list[str]], operations:list[str], timeframe:list[str], logger:logging.Logger) -> pd.DataFrame:
    logger.info("-> Statistical")

    for subset in groupby_subsets:

        buffer_list = operations.copy()

        logger.info(f"  -> {subset}")

        initials = "_".join([i[0] for i in subset])

        if 'sum' in buffer_list and 'cumsum' in buffer_list:
            df2 = df.groupby(timeframe+subset, as_index=False).agg({feature:'sum'})

            col_name1 = f"Sum{feature}_"+initials
            col_name2 = f"Cum{feature}_"+initials

            df2.rename(columns={feature:col_name1}, inplace=True)

            df2[col_name2] = df2.groupby(subset)[col_name1].cumsum()

            df = df.merge(df2[timeframe+[col_name1,col_name2]+subset], on=timeframe+subset, how="left")

            buffer_list.remove("sum")
            buffer_list.remove("cumsum")

        if len(buffer_list) >0:
            for operation in buffer_list:
                op = operation.capitalize()
                col_name = f"{op}{feature}_"+initials
                df[col_name] = df.groupby(timeframe+subset)[feature].transform(operation)
        
    return df



# Trend features
def add_trend_ft(df:pd.DataFrame, groupby_subset:list[str], logger:logging.Logger) -> pd.DataFrame:
    logger.info("-> Trend")
    df["GlobalTrend"] = df.groupby(groupby_subset).cumcount()+1
    df["IntraMonthTrend"] = df.groupby(groupby_subset+["Year","Month"]).cumcount()+1

    df.reset_index(drop=True, inplace=True)

    return df



if __name__ == "__main__":
    PATH = r"..\..\data\raw\raw_data.parquet"
    df_init = pd.read_parquet(PATH)
    df = df_init.copy()
    
    df.sort_values(by=["Client","Warehouse","Product","Date"], inplace=True)

    df = add_temporal_ft(df)
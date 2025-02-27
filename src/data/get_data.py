import pandas as pd
import numpy as np
import datetime as dt
import pathlib
from meteostat import Point, Daily

import warnings
warnings.filterwarnings("ignore")

CITIES = {
    'Beijing': (39.9042, 116.4074),
    'Shanghai': (31.2304, 121.4737),
    'Berlin': (52.5200, 13.4050),
    'Brussels': (50.8503, 4.3517),
    'Paris': (48.8566, 2.3522),
    'London': (51.5074, -0.1278),
    'Amsterdam': (52.3676, 4.9041),
    'Milan': (45.4642, 9.1900),
    'New York': (40.7128, -74.0060),
    'Chicago': (41.8781, -87.6298),
    'Miami': (25.7617, -80.1918),
    'Los Angeles': (34.0522, -118.2437),
    'Houston': (29.7604, -95.3698)
}

AVG_SUN_HOURS_PER_MONTH = [8.53, 9.9, 11.74, 13.92, 15.87, 15.9, 16.2, 14.86, 12.89, 10.55, 10.03, 7.85]

def _set_first_day_of_week(dates_series):
    return pd.to_datetime(dates_series).to_period('W').start_time


def get_first_monday_of_month(date):
    # Commencer par le premier jour du mois
    first_day_of_month = date.replace(day=1)
    # Calculer le jour de la semaine (0 = lundi, ..., 6 = dimanche)
    first_weekday = first_day_of_month.weekday()
    # Si le premier jour du mois est déjà un lundi (0), on le garde
    if first_weekday == 0:
        return first_day_of_month
    else:
        # Sinon, on ajoute le nombre de jours nécessaires pour atteindre le premier lundi
        days_to_add = 7 - first_weekday
        return first_day_of_month + pd.Timedelta(days=days_to_add)


def _get_sun_hours(start_date:str, end_date:str) -> pd.DataFrame:

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    adjusted_start_date = 0

    if start_date.month == 1:
        adjusted_start_date = start_date.replace(year=start_date.year - 1, month=12, day=15)
    else:
        adjusted_start_date = start_date.replace(month=start_date.month - 1, day=15)
    
    if end_date.month == 12:
        adjusted_end_date = end_date.replace(year=end_date.year + 1, month=1, day=15)
    else:
        adjusted_end_date = end_date.replace(month=end_date.month + 1, day=15)

    monthly_dates = pd.date_range(start=adjusted_start_date, end=adjusted_end_date, freq='MS')

    df = pd.DataFrame({"Date":monthly_dates})

    df["Date"] = pd.to_datetime(df["Date"])

    sun_dict = {index+1 : value for index,value in enumerate(AVG_SUN_HOURS_PER_MONTH)}

    df["Sun"] = df["Date"].dt.month.map(sun_dict)

    date_range = pd.date_range(start=adjusted_start_date, end=adjusted_end_date, freq='D')
    
    df_daily = df.set_index("Date").reindex(date_range)
    
    df_daily["Sun"] = df_daily["Sun"].interpolate(method='linear')

    df_daily.dropna(subset="Sun", inplace=True)

    df_daily = df_daily.reset_index().rename(columns={'index': "Date"})

    df_weekly = df_daily.resample('W-Mon', on="Date").mean()

    df_weekly = df_weekly.reset_index()

    df_weekly = df_weekly[(df_weekly.Date <= end_date) & (df_weekly.Date >= start_date)]

    return df_weekly


def _get_city_weather(city_name:str, lat:float, lon:float, start_date:str, end_date:str) -> pd.DataFrame:

    location = Point(lat, lon)
   
    start = dt.datetime.strptime(start_date, "%Y-%m-%d")

    end = dt.datetime.strptime(end_date, "%Y-%m-%d")

    df = Daily(location, start, end)
    df = df.fetch()

    df["Date"] = _set_first_day_of_week(df.index)

    weekly_data = df.groupby(["Date"]).agg({'tavg': 'mean',   # Température moyenne (°C)
                                            'prcp': 'sum'     # Pluviométrie totale (mm)
                                            }).reset_index()

    weekly_data['City'] = city_name

    return weekly_data


def get_weather_data(start_date:str, end_date:str):
    df_city_weather = pd.DataFrame()

    for city, (lat, lon) in CITIES.items():
        
        city_weather = _get_city_weather(city, lat, lon, start_date, end_date)
        df_city_weather = pd.concat([df_city_weather, city_weather])

    df_city_weather = df_city_weather.groupby(["Date"]).agg({'tavg': 'mean',
                                                   'prcp': 'sum'
                                                   }).reset_index()
    
    df_city_weather.rename(columns= {'tavg':"Temp_celcius", 'prcp':"Rain_mm"}, inplace=True)
    df_city_weather = df_city_weather.loc[:df_city_weather.index[-2],:]

    df_sun = _get_sun_hours(start_date, end_date)

    df_weather = pd.merge(df_city_weather, df_sun, on="Date", how="left")
    df_weather.dropna(subset="Sun", inplace=True)

    df_weather.Temp_celcius = df_weather.Temp_celcius.round(1)
    df_weather.Rain_mm = df_weather.Rain_mm.round()
    df_weather.Rain_mm.astype(np.int16)
    df_weather.Sun = df_weather.Sun.round(1)

    return df_weather


def get_game_data(folder_path:str, feature:str) -> pd.DataFrame:

    folder_path = pathlib.Path(folder_path).joinpath(feature)
    dataframes = []

    for id,file in enumerate(folder_path.glob("*.csv")):
        df = pd.read_csv(file)

        df = df.set_index(["Client","Warehouse","Product"], drop=True)

        df = pd.DataFrame(df.stack(future_stack=True))

        df.index.rename(["Client","Warehouse","Product","Date"], inplace=True)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        df.columns = ["Client","Warehouse","Product","Date",f"{feature}"]
        df["Phase"] = id+1

        df.Date = pd.to_datetime(df.Date)

        df.Client = df.Client.astype(np.int8)
        df.Warehouse = df.Warehouse.astype(np.int16)
        df.Product = df.Product.astype(np.int16)
        df[f"{feature}"] = df[f"{feature}"].astype(np.float32)
        df.Phase = df.Phase.astype(np.int8)

        df = df.fillna(0)

        dataframes.append(df)
    
    df_concat = pd.concat(dataframes, axis=0)

    return df_concat


def get_covid_data(path:str) -> pd.DataFrame:

    df_covid = pd.read_csv(path)

    df_covid["Date_reported"] = pd.to_datetime(df_covid["Date_reported"])
    df_covid["Date_reported"] = df_covid["Date_reported"] - pd.Timedelta(days=6)

    df_covid.rename(columns={"Date_reported":"Date"}, inplace=True)

    WHOREGION = ["EURO","SEARO","WPRO","EMRO"]

    COUNTRIES = ["US","CA","MX"]

    df_covid = df_covid.loc[(df_covid["WHO_region"].isin(WHOREGION)) | (df_covid["Country_code"].isin(COUNTRIES))]

    df_covid = df_covid.groupby(["Date"]).agg({"Cumulative_cases": "max",
                                               "Cumulative_deaths": "max"
                                               }).reset_index()

    df_covid["Cases"] = (df_covid["Cumulative_cases"] - df_covid["Cumulative_cases"].shift(1)).fillna(df_covid["Cumulative_cases"])
    df_covid["Deaths"] = (df_covid["Cumulative_deaths"] - df_covid["Cumulative_deaths"].shift(1)).fillna(df_covid["Cumulative_deaths"])

    df_covid["Cases"] = df_covid["Cases"].astype(int)
    df_covid["Deaths"] = df_covid["Deaths"].astype(int)

    return df_covid


def get_economic_data(path:str) -> pd.DataFrame:
    path = pathlib.Path(path)
    feature_name = path.stem
   
    df = pd.read_csv(path)

    col_list = df.columns
    
    df.DATE = pd.to_datetime(df.DATE)
    df.sort_values(by="DATE", inplace=True)

    df["DATE"] = df["DATE"].apply(get_first_monday_of_month)

    weekly_dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq='W-MON')
    df2 = pd.DataFrame({"DATE":weekly_dates})

    df = pd.merge(df2,df, on=["DATE"], how="left")
    df[f"{feature_name}"] = df[f"{col_list[1]}"].interpolate(method='linear')

    df.drop([f"{col_list[1]}"], axis=1, inplace=True)
    df.columns = ["Date",f"{feature_name}"]
    df[f"{feature_name}"] = df[f"{feature_name}"].round(3)

    return df


def get_future_dataframe(df:pd.DataFrame, start_date:dt.date, end_date:dt.date) -> pd.DataFrame:

    final_range = pd.date_range(start_date, end_date, freq="W-MON", inclusive="both")
    final_range = [i.date() for i in final_range]

    df_final = df[["Client","Warehouse","Product"]].copy()
    df_final = df_final.drop_duplicates()
    df_final = df_final.sort_values(by=["Client","Warehouse","Product"])

    df_final.set_index(["Client","Warehouse","Product"], drop=True, inplace=True)

    df_final[final_range] = np.nan

    df_final = pd.DataFrame(df_final.stack(future_stack=True))
    df_final.sort_index(inplace=True)
    df_final.reset_index(inplace=True)
    df_final.columns = ["Client","Warehouse","Product","Date","y_pred"]
    df_final.drop("y_pred", axis=1, inplace=True)

    return df_final


if __name__ == "__main__":
    PATH= r"..\..\data\0_external"
    PATH_covid = r"..\..\data\0_external\WHO-COVID-19-global-data.csv"
    PATH_GEPU = r"..\..\data\0_external\GEPU_world.csv"
    PATH_CPI = r"..\..\data\0_external\CPI_G7.csv"


# Create Sales & Price dataset
    df_sales = get_game_data(PATH, "Sales")
    df_sales.to_parquet(r"..\..\data\1_serialized\sales.parquet", index=False)
    df_price = get_game_data(PATH, "Price")
    df_price.to_parquet(r"..\..\data\1_serialized\price.parquet", index=False)
    
# Create Weather dataset
    df_weather = get_weather_data("2019-12-20", "2024-06-15")
    df_weather.to_parquet(r"..\..\data\1_serialized\weather.parquet", index=False)

# Create Covid dataset
    df_covid = get_covid_data(PATH_covid)
    df_covid.to_parquet(r"..\..\data\1_serialized\covid.parquet", index=False)


# Create economic dataset
    df_GEPU = get_economic_data(PATH_GEPU)
    df_GEPU.to_parquet(r"..\..\data\1_serialized\GEPU.parquet", index=False)
    # The Global Economic Policy Uncertainty Index is a GDP-weighted average of national EPU indices for 20 countries:
    # Australia, Brazil, Canada, Chile, China, France, Germany, Greece, India,
    # Ireland, Italy, Japan, Mexico, the Netherlands, Russia,
    # South Korea, Spain, Sweden, the United Kingdom, and the United States..

    # Baker, Scott R., Bloom, Nick and Davis, Stephen J., Global Economic Policy Uncertainty Index: Current Price Adjusted GDP [GEPUCURRENT], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/GEPUCURRENT, September 20, 2024.
    

    df_CPI = get_economic_data(PATH_CPI)
    df_CPI.to_parquet(r"..\..\data\1_serialized\CPI.parquet", index=False)
    # Consumer Price Index: All items: Total: Total for G7
    # Organization for Economic Co-operation and Development, Consumer Price Index: All items: Total: Total for G7 [G7CPALTT01GPM], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/G7CPALTT01GPM, September 20, 2024.




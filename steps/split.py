import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


class TimeSerieSpliter:

    def __init__(self, df:pd.DataFrame, time_feature:str) -> None:
        self.df = df
        self.time_feature = time_feature
        self.ts_splits = None


    def time_serie_split(self, predict_week_range:int, splits:int=5) -> list[tuple]:
        """
        Returns: list[tuple]: [(train_idx, val_idx), (train_idx, val_idx), ... ]
        """
        
        splits_list = []
        date_max = self.df[self.time_feature].max()

        for i in range(splits):
            left_bound = date_max - dt.timedelta(weeks= (splits-i) * predict_week_range)
            right_bound = date_max - dt.timedelta(weeks= (splits-i-1) * predict_week_range)

            train_idx = self.df.loc[(self.df[self.time_feature] <= left_bound)].index
            val_idx = self.df.loc[(self.df[self.time_feature] > left_bound) & (self.df[self.time_feature] <= right_bound)].index

            splits_list.append((train_idx,val_idx))

        self.ts_splits = None
        self.ts_splits = splits_list

        return self.ts_splits
    

    def plot_splits(self, tracking_feature:str, agg_operation:str="sum") -> None:
        fig, axs = plt.subplots(len(self.ts_splits),1, figsize=(15,15), sharex=True)
        fold = 0
        for train_idx, val_idx in self.ts_splits:
            train = self.df.iloc[train_idx].copy()
            val = self.df.iloc[val_idx].copy()

            train = train.groupby(self.time_feature).agg(Sales = (tracking_feature, agg_operation))
            val = val.groupby(self.time_feature).agg(Sales = (tracking_feature, agg_operation))

            if len(self.ts_splits)==1:
                train.Sales.plot(ax=axs,label="Training Set",title=f"{fold=}")
                val.Sales.plot(ax=axs,label="Test Set")

                axs.axvline(val.index.min(), color='black', ls='--')

            else:
                train.Sales.plot(ax=axs[fold],label="Training Set",title=f"{fold=}")
                val.Sales.plot(ax=axs[fold],label="Test Set")

                axs[fold].axvline(val.index.min(), color='black', ls='--')
            fold += 1
        plt.show()





if __name__ == "__main__":

    df = pd.read_parquet(r"data\4_raw\raw_data.parquet")

    df.sort_values(by=["Date","Client","Warehouse","Product"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    ts_spliter = TimeSerieSpliter(df=df, time_feature="Date")

    ts_splits = ts_spliter.time_serie_split(predict_week_range=13, splits=1)

    ts_spliter.plot_splits(tracking_feature="Sales", agg_operation="sum")
import numpy as np
import pandas as pd


def custom_score(y_test:np.ndarray, y_pred:np.ndarray) -> float:
    y_diff = y_pred - y_test
    abs_err = np.nansum(abs(y_diff))
    err = np.nansum(y_diff)
    total_error = abs_err + abs(err)

    total_actual = np.nansum(y_test)
    if total_actual == 0:
        raise ZeroDivisionError("sum(y_test) == 0, cannot normalize the score.")

    normalized_score = total_error / total_actual

    score = round(normalized_score * 100, 1)
    
    return score

def custom_score_statsforecast(df: pd.DataFrame,
                               models: list[str],
                               id_col: str = "unique_id",
                               target_col: str = "y",
                               ) -> pd.DataFrame:


    y_diff = df[models].sub(df[target_col], axis=0)
    
    abs_err = y_diff.abs().sum(axis=1)
    err = y_diff.sum(axis=1) 
    total_error = abs_err + err.abs()

    total_actual = df.groupby(id_col, observed=True)[target_col].transform('sum')
    
    if (total_actual == 0).any():
        raise ZeroDivisionError("sum(y_test) == 0 in one or more groups, cannot normalize the score.")

    normalized_score = total_error / total_actual

    score = (
        normalized_score.groupby(df[id_col], observed=True)
        .mean()
        .reset_index(name="custom_score")
    )
    
    return score

import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

def oos_r2(y_true, y_forecast):
    """
    Campbell-Thompson OOS R^2.
    Benchmark = expanding-window historical mean, lagged by 1 period (standard).
    """
    valid = ~np.isnan(y_forecast)
    idx = np.where(valid)[0]
    
    # Expanding mean lagged by 1: at time t, benchmark = mean(y[0],...,y[t-1])
    y_condmean = np.divide(y_true.cumsum(), (np.arange(len(y_true)) + 1))
    y_condmean = np.insert(y_condmean, 0, np.nan)[:-1]  # lag by one period
    
    ss_res = np.nansum((y_true[valid] - y_forecast[valid]) ** 2)
    ss_tot = np.nansum((y_true[valid] - y_condmean[valid]) ** 2)
    
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

def expanding_window(model_class, X, y, dates, oos_start, gap=0):
    """
    Expanding-window OOS forecasting.
    
    Parameters
    ----------
    model_class : model with fit/predict interface
    X : pd.DataFrame, features
    y : np.array, target
    dates : pd.DatetimeIndex
    oos_start : pd.Timestamp
    h : int, forecast horizon (default 12)
    gap : int, number of periods to gap between training and prediction.
          gap=0  : original Bianchi (train on y[:t], predict y[t])
          gap=12 : corrigendum (train on y[:t-12], predict y[t])
                   ensures no overlapping return in training uses
                   future information relative to prediction time.
    
    Returns
    -------
    np.array of forecasts (NaN where no forecast is made)
    """
    y_forecast = np.full(len(y), np.nan)
    oos_indices = np.where(dates >= oos_start)[0]

    for t in tqdm(oos_indices):
        model = deepcopy(model_class)
        
        train_end = t - gap  # with gap=0: train on [:t], with gap=12: train on [:t-12]
        
        if train_end < 2:  # need at least 2 training observations
            continue
            
        X_train = X.iloc[:train_end]
        y_train = y[:train_end]

        model.fit(X_train, y_train)
        y_forecast[t] = model.predict(X.iloc[[t]])

    return y_forecast

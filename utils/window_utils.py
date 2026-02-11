# import numpy as np

# def oos_r2(y_true, y_forecast):
#     """
#     Campbell-Thompson OOS R^2.
#     Benchmark = expanding-window historical mean (computed at each t).
#     """
#     valid = ~np.isnan(y_forecast)
#     y_t = y_true[valid]
#     y_f = y_forecast[valid]

#     # Expanding mean benchmark: at each OOS step i, mean of y_true up to that point
#     # We need the cumulative mean of y_true *before* each OOS observation
#     # y_true indices correspond to the same positions, so reconstruct from y_true
#     idx = np.where(valid)[0]
#     y_mean = np.array([np.mean(y_true[:j]) for j in idx])

#     ss_res = np.sum((y_t - y_f) ** 2)
#     ss_tot = np.sum((y_t - y_mean) ** 2)
#     return 1 - ss_res / ss_tot


# def expanding_window(model_class, X, y, dates, oos_start):
#     """
#     Expanding-window OOS forecasting.

#     y_{t+1} = f(X_t), so we shift: target at t+1 is paired with features at t.

#     Parameters
#     ----------
#     model_class : object with .fit(X, y) and .predict(X) methods
#     X : np.ndarray (T, K)     features
#     y : np.ndarray (T,)       response
#     dates : pd.DatetimeIndex   index aligned with X and y
#     oos_start : pd.Timestamp   first OOS date (date of the *target* y_{t+1})

#     Returns
#     -------
#     y_forecast : np.ndarray (T,)   OOS forecasts (NaN where not forecasted)
#     """
#     T = len(y)
#     y_forecast = np.full(T, np.nan)

#     # Align: X_t predicts y_{t+1}
#     # So training pairs are (X[0], y[1]), (X[1], y[2]), ..., (X[t-1], y[t])
#     # At time t we train on X[:t] -> y[1:t+1], then predict y[t+1] from X[t]

#     # Find first OOS index (index of the target y_{t+1})
#     oos_indices = np.where(dates >= oos_start)[0]
#     for t_plus_1 in oos_indices:
#         # t_plus_1 is the index of y_{t+1}

#         # Training data: X[0:t] predicts y[1:t+1]  (where t = t_plus_1 - 1)
#         t = t_plus_1 - 1
#         X_train = X[:t]       # X[0], X[1], ..., X[t-1]
#         y_train = y[1:t+1]    # y[1], y[2], ..., y[t]

#         # Fit model
#         model = model_class.__class__()  # fresh instance
#         model.fit(X_train, y_train)

#         # Predict y_{t+1} from X[t]
#         X_curr = X[t].reshape(1, -1)
#         y_forecast[t_plus_1] = model.predict(X_curr).item()

#     return y_forecast

import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

def oos_r2(y_true, y_forecast):
    """
    Campbell-Thompson OOS R^2.
    Benchmark = expanding-window historical mean (computed at each t).
    """
    valid = ~np.isnan(y_forecast)
    y_t = y_true[valid]
    y_f = y_forecast[valid]

    # Expanding mean benchmark: at each OOS step i, mean of y_true up to that point
    # We need the cumulative mean of y_true *before* each OOS observation
    # y_true indices correspond to the same positions, so reconstruct from y_true
    idx = np.where(valid)[0]
    y_mean = np.array([np.mean(y_true[:j]) for j in idx])

    ss_res = np.sum((y_t - y_f) ** 2)
    ss_tot = np.sum((y_t - y_mean) ** 2)
    return 1 - ss_res / ss_tot


# def expanding_window(model_class, X, y, dates, oos_start):
#     """
#     Expanding-window OOS forecasting.

#     y_{t+1} = f(X_t), so we shift: target at t+1 is paired with features at t.

#     Parameters
#     ----------
#     model_class : object with .fit(X, y) and .predict(X) methods (initialized instance)
#     X : np.ndarray or pd.DataFrame (T, K)     features
#     y : np.ndarray (T,)       response
#     dates : pd.DatetimeIndex   index aligned with X and y
#     oos_start : pd.Timestamp   first OOS date (date of the *target* y_{t+1})

#     Returns
#     -------
#     y_forecast : np.ndarray (T,)   OOS forecasts (NaN where not forecasted)
#     """
#     T = len(y)
#     y_forecast = np.full(T, np.nan)

#     # Align: X_t predicts y_{t+1}
#     # So training pairs are (X[0], y[1]), (X[1], y[2]), ..., (X[t-1], y[t])
#     # At time t we train on X[:t] -> y[1:t+1], then predict y[t+1] from X[t]

#     # Find first OOS index (index of the target y_{t+1})
#     oos_indices = np.where(dates >= oos_start)[0]
#     for t_plus_1 in tqdm(oos_indices, desc="Expanding window OOS"):
#         # t_plus_1 is the index of y_{t+1}

#         # Training data: X[0:t] predicts y[1:t+1]  (where t = t_plus_1 - 1)
#         t = t_plus_1 - 1
        
#         # Handle DataFrame vs Numpy slicing
#         if isinstance(X, pd.DataFrame):
#             X_train = X.iloc[:t]
#             # Use [[t]] (double brackets) to return a 1-row DataFrame 
#             # This preserves columns/MultiIndex, allowing models to select X['forwards']
#             X_curr = X.iloc[[t]]
#         else:
#             X_train = X[:t]
#             X_curr = X[t].reshape(1, -1)

#         y_train = y[1:t+1]

#         # Fit model
#         # Use deepcopy to preserve hyperparameters (e.g. alpha) from the passed instance
#         model = deepcopy(model_class)
#         model.fit(X_train, y_train)

#         # Predict y_{t+1} from X[t]
#         pred = model.predict(X_curr)
        
#         # Handle different return types (scalar, 1-element array/series)
#         if hasattr(pred, 'item'):
#             y_forecast[t_plus_1] = pred.item()
#         else:
#             y_forecast[t_plus_1] = pred

#     return y_forecast

def expanding_window(model_class, X, y, dates, oos_start, h=12):
    """
    Expanding-window OOS forecasting with proper handling of
    holding-period overlap.

    rx[t] is initiated at t but realized at t+h. So at prediction time t,
    the last realized return is y[t-h].

    Training pairs: (X[s], y[s+1]) for s where y[s+1] is realized,
    i.e. s+1+h <= t, i.e. s <= t-h-1.

    Parameters
    ----------
    model_class : object with .fit(X, y) and .predict(X) methods (initialized instance)
    X : np.ndarray or pd.DataFrame (T, K)     features
    y : np.ndarray (T,)       response
    dates : pd.DatetimeIndex   index aligned with X and y
    oos_start : pd.Timestamp   first OOS date (date of the *target* y_{t+1})
    h : int                    holding period in months (default 12)

    Returns
    -------
    y_forecast : np.ndarray (T,)   OOS forecasts (NaN where not forecasted)
    """
    T = len(y)
    y_forecast = np.full(T, np.nan)

    oos_indices = np.where(dates >= oos_start)[0]
    for t_plus_1 in tqdm(oos_indices, desc="Expanding window OOS"):
        t = t_plus_1 - 1

        # Last realized return index: y[t-h+1] is realized at t+1... no.
        # y[s] is realized at s+h. At time t, realized returns: y[s] where s+h <= t, i.e. s <= t-h
        last_realized = t - h

        if last_realized < 1:
            continue  # not enough data

        # Training: (X[s], y[s+1]) for s = 0, ..., last_realized-1
        # So X_train = X[0:last_realized], y_train = y[1:last_realized+1]
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[:last_realized]
            X_curr = X.iloc[[t]]
        else:
            X_train = X[:last_realized]
            X_curr = X[t].reshape(1, -1)

        y_train = y[1:last_realized + 1]

        model = deepcopy(model_class)
        model.fit(X_train, y_train)

        pred = model.predict(X_curr)

        if hasattr(pred, 'item'):
            y_forecast[t_plus_1] = pred.item()
        else:
            y_forecast[t_plus_1] = pred

    return y_forecast
import numpy as np

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


def expanding_window(model_class, X, y, dates, oos_start):
    """
    Expanding-window OOS forecasting.

    y_{t+1} = f(X_t), so we shift: target at t+1 is paired with features at t.

    Parameters
    ----------
    model_class : object with .fit(X, y) and .predict(X) methods
    X : np.ndarray (T, K)     features
    y : np.ndarray (T,)       response
    dates : pd.DatetimeIndex   index aligned with X and y
    oos_start : pd.Timestamp   first OOS date (date of the *target* y_{t+1})

    Returns
    -------
    y_forecast : np.ndarray (T,)   OOS forecasts (NaN where not forecasted)
    """
    T = len(y)
    y_forecast = np.full(T, np.nan)

    # Align: X_t predicts y_{t+1}
    # So training pairs are (X[0], y[1]), (X[1], y[2]), ..., (X[t-1], y[t])
    # At time t we train on X[:t] -> y[1:t+1], then predict y[t+1] from X[t]

    # Find first OOS index (index of the target y_{t+1})
    oos_indices = np.where(dates >= oos_start)[0]
    for t_plus_1 in oos_indices:
        # t_plus_1 is the index of y_{t+1}

        # Training data: X[0:t] predicts y[1:t+1]  (where t = t_plus_1 - 1)
        t = t_plus_1 - 1
        X_train = X[:t]       # X[0], X[1], ..., X[t-1]
        y_train = y[1:t+1]    # y[1], y[2], ..., y[t]

        # Fit model
        model = model_class.__class__()  # fresh instance
        model.fit(X_train, y_train)

        # Predict y_{t+1} from X[t]
        X_curr = X[t].reshape(1, -1)
        y_forecast[t_plus_1] = model.predict(X_curr).item()

    return y_forecast
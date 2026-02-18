import numpy as np
from copy import deepcopy
from tqdm import tqdm

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

    if y.ndim == 1:
        y_forecast = np.full(len(y), np.nan)
    else:
        y_forecast = np.full(y.shape, np.nan)

    oos_indices = np.where(dates >= oos_start)[0]

    for t in tqdm(oos_indices):
        model = deepcopy(model_class)
        
        train_end = t - gap  # with gap=0: train on [:t], with gap=12: train on [:t-12]
        
        if train_end < 2:  # need at least 2 training observations
            continue
            
        X_train = X.iloc[:train_end]
        y_train = y[:train_end]

        model.fit(X_train, y_train)
        pred = model.predict(X.iloc[[t]])

        if y.ndim == 1:
            y_forecast[t] = pred
        else:
            y_forecast[t, :] = pred.flatten()

    return y_forecast


def oos_r2(y_true, y_forecast, benchmark='hist_mean', gap=0, **kwargs):
    """
    Campbell-Thompson OOS R^2 with selectable benchmark.
    Supports single-output (T,) or multi-output (T, n_outputs).
    For multi-output, returns an array of R^2 values, one per output.
    
    Parameters
    ----------
    y_true : np.array
    y_forecast : np.array
    benchmark : str, one of:
        'hist_mean'   - expanding-window historical mean (default, Campbell-Thompson)
        'ewma'        - exponentially weighted moving average (specify `halflife` in kwargs)
        'rolling'     - rolling window mean (specify `window` in kwargs)
        'ar1'         - expanding-window AR(1)
        'zero'        - constant zero forecast (pure EH null: no excess return)
    **kwargs : additional parameters for the benchmark
        halflife : int, EWMA half-life in periods (default 60)
        window   : int, rolling mean window in periods (default 60)
    
    Returns
    -------
    float : OOS R^2
    """
    if y_true.ndim == 2:
        n_outputs = y_true.shape[1]
        return np.array([
            oos_r2(y_true[:, i], y_forecast[:, i], benchmark=benchmark, **kwargs)
            for i in range(n_outputs)
        ])

    valid = ~np.isnan(y_forecast)
    
    if benchmark == 'hist_mean':
        # At time t, with gap=g, only y[0],...,y[t-g-1] are realized
        y_bench = np.full(len(y_true), np.nan)
        for t in range(1, len(y_true)):
            end = t - gap if gap > 0 else t
            if end < 1:
                continue
            y_bench[t] = np.mean(y_true[:end])
    
    elif benchmark == 'ewma':
        halflife = kwargs.get('halflife', 60)
        alpha = 1 - np.exp(-np.log(2) / halflife)
        y_bench = np.full(len(y_true), np.nan)
        ewma = y_true[0]
        for t in range(1, len(y_true)):
            y_bench[t] = ewma  # forecast made before observing y[t]
            ewma = alpha * y_true[t] + (1 - alpha) * ewma
    
    elif benchmark == 'rolling':
        window = kwargs.get('window', 60)
        y_bench = np.full(len(y_true), np.nan)
        for t in range(1, len(y_true)):
            start = max(0, t - window)
            y_bench[t] = np.mean(y_true[start:t])
    
    elif benchmark == 'ar1':
        y_bench = np.full(len(y_true), np.nan)
        for t in range(2, len(y_true)):
            y_train = y_true[:t]
            # OLS: y_t = a + b * y_{t-1}
            x = np.column_stack([np.ones(t - 1), y_train[:-1]])
            coeffs = np.linalg.lstsq(x, y_train[1:], rcond=None)[0]
            y_bench[t] = coeffs[0] + coeffs[1] * y_true[t - 1]
    
    elif benchmark == 'zero':
        y_bench = np.zeros(len(y_true))
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    # Mask benchmark NaNs as well
    valid = valid & ~np.isnan(y_bench)
    
    ss_res = np.nansum((y_true[valid] - y_forecast[valid]) ** 2)
    ss_tot = np.nansum((y_true[valid] - y_bench[valid]) ** 2)
    
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm

def expanding_window(model_class, X, y, dates, oos_start, 
                     gap=0, refit_freq=1, coef_callback=None,
                     save_callback=None,
                     progress=True,
                     tqdm_position=None,
                     tqdm_desc="expanding window",
                     tqdm_leave=False):
    """
    Unified Forecasting Engine.
    """
    if y.ndim == 1:
        y_forecast = np.full(len(y), np.nan)
    else:
        y_forecast = np.full(y.shape, np.nan)

    oos_indices = np.where(dates >= oos_start)[0]
    model = None

    iterator = oos_indices
    if progress:
        iterator = tqdm(
            oos_indices,
            desc=tqdm_desc,
            position=tqdm_position,
            leave=tqdm_leave,
            dynamic_ncols=True
        )

    for i, t in enumerate(iterator):
        if i % refit_freq == 0:
            if model is None:
                current_model = deepcopy(model_class)
            else:
                current_model = deepcopy(model)

            train_end = t - gap
            X_train, y_train = X.iloc[:train_end], y[:train_end]
            current_model.fit(X_train, y_train)
            model = current_model

            if save_callback is not None:
                save_callback(model=current_model, refit_i=i, t_index=t, date_value=dates[t])

            if coef_callback is not None and hasattr(model, "model"):
                try:
                    coef_callback(current_model.model.coef_)
                except AttributeError:
                    print("Warning: Model does not have .model.coef_ attribute for callback.")

        # Use sequence architectures' own prediction method if available
        if hasattr(model, "predict_at"):
            pred = model.predict_at(X, t)
        else:
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
        'zero'        - constant zero forecast (pure EH null: no excess return, GKX benchmark)
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

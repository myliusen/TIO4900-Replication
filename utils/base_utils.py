import os
import pandas as pd
from typing import Tuple
import numpy as np

# Repo root: one level up from utils/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_fred_md_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load FRED-MD dataset with proper handling of transformation codes
    
    Parameters:
    -----------
    filepath : str
        Path to FRED-MD CSV file
        
    Returns:
    --------
    tuple
        Raw data and transformation codes
    """
    # Read the full file
    full_data = pd.read_csv(filepath)
    
    # Extract transformation codes (second row)
    transform_codes = full_data.iloc[0, 1:].astype(int)
    transform_codes.name = 'transform_codes'
    
    # Extract data (third row onwards)
    data = full_data.iloc[1:].copy()
    data = data.reset_index(drop=True)
    
    # Convert date column
    data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
    data = data.set_index('date')

    # align dates to last date in previous month
    data.index = pd.to_datetime(data.index)  # ensure datetime index
    def _prev_month_end(dt):
        if pd.isna(dt):
            return dt
        return (dt.replace(day=1) - pd.Timedelta(days=1)).normalize()
    data.index = pd.DatetimeIndex([_prev_month_end(d) for d in data.index])
    data.index.name = 'date'

    # Convert all other columns to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data, transform_codes


def apply_fred_transformations(data: pd.DataFrame, transform_codes: pd.Series) -> pd.DataFrame:
    """
    Apply FRED-MD transformation codes to data
    
    Transform codes:
    1 = no transformation (levels)
    2 = first difference
    3 = second difference  
    4 = log
    5 = log first difference
    6 = log second difference
    7 = delta(x_t/x_{t-1} - 1)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data with variables in columns
    transform_codes : pd.Series
        Transformation codes for each variable
        
    Returns:
    --------
    pd.DataFrame
        Transformed data
    """
    transformed = data.copy()
    
    for col in data.columns:
        if col not in transform_codes.index:
            continue
            
        code = transform_codes[col]
        series = data[col].copy()
        
        # Handle missing values
        if series.isna().all():
            continue
            
        try:
            if code == 1:  # Levels
                transformed[col] = series
            elif code == 2:  # First difference
                transformed[col] = series.diff()
            elif code == 3:  # Second difference
                transformed[col] = series.diff().diff()
            elif code == 4:  # Log
                # Only take log of positive values
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series)
                else:
                    transformed[col] = np.nan
            elif code == 5:  # Log first difference
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series).diff()
                else:
                    transformed[col] = np.nan
            elif code == 6:  # Log second difference
                series_pos = series[series > 0]
                if len(series_pos) > 0:
                    transformed[col] = np.log(series).diff().diff()
                else:
                    transformed[col] = np.nan
            elif code == 7:  # Delta(x_t/x_{t-1} - 1)
                transformed[col] = (series / series.shift(1) - 1).diff()
            else:
                print(f"Unknown transformation code {code} for variable {col}")
                transformed[col] = series
                
        except Exception as e:
            print(f"Error transforming {col} with code {code}: {e}")
            transformed[col] = np.nan
            
    return transformed


def get_fred_data(filepath: str, start: str, end: str) -> pd.DataFrame:
    """Convenience function to load and transform FRED-MD data in one step. """
    # Resolve relative paths against repo root
    if not os.path.isabs(filepath):
        filepath = os.path.join(_REPO_ROOT, filepath)
    fred_md = apply_fred_transformations(*load_fred_md_data(filepath))
    return fred_md.loc[start:end]


def get_yields(type: str, start: str, end: str, maturities: list) -> pd.DataFrame:
    """Load and preprocess KR yields data."""
    if type == 'kr':
        yields = pd.read_csv(os.path.join(_REPO_ROOT, 'data', 'yield_panel_monthly_frequency_monthly_maturity.csv'), index_col=0, parse_dates=True)
        yields.index.name = 'date'
    if type == "lw":
        pass # Implement loading for LW yields if needed

    yields = yields[maturities]
    
    # Snap business-day month-ends to calendar month-ends (e.g. Sep 29 -> Sep 30)
    yields.index = yields.index + pd.offsets.MonthEnd(0)
    yields.index.name = 'date'

    return yields.loc[start:end]


def get_forward_rates(yields: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forward rates from zero-coupon yields.
    
    f_t(n) = log P_t(n-1) - log P_t(n)
    
    With monthly maturities m (in months), n = m/12 in years:
        log P_t(m) = -(m/12) * y_t(m)
        f_t(m) = -(m-1)/12 * y_t(m-1) + m/12 * y_t(m)
    
    Forward rates are computed for maturities > 1 month, i.e. 12, 24, ..., 120.
    For these maturities, m-1 means one year shorter (m-12), matching the
    yearly loan interpretation: forward rate for a loan from t+n-1 to t+n years.
    
    Parameters:
    -----------
    yields : pd.DataFrame
        Zero-coupon yields with monthly maturity columns (as strings: '1','12','24',...,'120')
        
    Returns:
    --------
    pd.DataFrame
        Forward rates for maturities 12, 24, ..., 120
    """
    # Maturities for which we compute forward rates (yearly maturities in months)
    forward_maturities = [str(i) for i in range(12, 121) if i % 12 == 0]
    
    forward_rates = pd.DataFrame(index=yields.index)
    
    for m_str in forward_maturities:
        m = int(m_str)
        # n = m/12 in years, n-1 = (m-12)/12 in years
        m_prev = m - 12  # maturity one year shorter (in months)
        
        # log P_t(m) = -(m/12) * y_t(m)
        log_p_m = -(m / 12) * yields[m_str]
        
        if m_prev == 0:
            # log P_t(0) = 0 (price of a matured bond is 1, log(1) = 0)
            log_p_m_prev = 0.0
        else:
            log_p_m_prev = -(m_prev / 12) * yields[str(m_prev)]
        
        # f_t(n) = log P_t(n-1) - log P_t(n)
        forward_rates[m_str] = log_p_m_prev - log_p_m
    
    return forward_rates


def get_excess_returns(yields: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annual excess holding-period returns from zero-coupon yields.
    
    r_{t+12}(m) = log P_{t+12}(m-12) - log P_t(m)
    rx_{t+12}(m) = r_{t+12}(m) - y_t(12)
    
    where m is maturity in months. The holding period is 12 months.
    Excess returns are computed for maturities 24, 36, ..., 120 months
    (since m-12 must be >= 12 for a meaningful bond to remain).
    
    Parameters:
    -----------
    yields : pd.DataFrame
        Zero-coupon yields with monthly maturity columns (as strings: '1','12','24',...,'120')
        
    Returns:
    --------
    pd.DataFrame
        Excess returns for maturities 24, 36, ..., 120 (indexed at t+12)
    """
    # Maturities for excess returns: need m >= 24 so that m-12 >= 12
    excess_maturities = [str(i) for i in range(24, 121) if i % 12 == 0]
    
    excess_returns = pd.DataFrame(index=yields.index)
    
    for m_str in excess_maturities:
        m = int(m_str)
        m_prev = m - 12  # maturity after holding for 12 months
        
        # log P_t(m) = -(m/12) * y_t(m)
        log_p_t_m = -(m / 12) * yields[m_str]
        
        # log P_{t+12}(m-12) = -((m-12)/12) * y_{t+12}(m-12)
        log_p_t12_mprev = -(m_prev / 12) * yields[str(m_prev)].shift(-12)
        
        # Holding-period return: r_{t+12}(m) = log P_{t+12}(m-12) - log P_t(m)
        hpr = log_p_t12_mprev - log_p_t_m
        
        # Risk-free rate over holding period: y_t(12) (the 12-month yield)
        rf = yields['12']
        
        # Excess return: rx_{t+12}(m) = r_{t+12}(m) - y_t(12)
        excess_returns[m_str] = hpr - rf
    
    return excess_returns

def get_fredmd_grouping():
    """
    Return (ordered_series, series_to_group) for the earlier FRED-MD style
    grouping used previously in LN.py (uppercase FRED-MD mnemonics).
    """
    ordered_series = [
        # Output & Income
        'RPI','W875RX1','INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD','IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222s','IPFUELS','NAPMPI','CUMFNS',
        # Labor Market
        'HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN','UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx','PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP','NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007','AWOTMAN','AWHMAN','NAPMEI','CES0600000008','CES2000000008','CES3000000008',
        # Consumption & Housing
        'HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW','PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW',
        # Orders & Inventories
        'DPCERA3M086SBEA','CMRMTSPLx','RETAILx','NAPM','NAPMNOI','NAPMSDI','NAPMII','ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx','UMCSENTx',
        # Money & Credit
        'M1SL','M2SL','M2REAL','AMBSL','TOTRESNS','NONBORRES','BUSLOANS','REALLN','NONREVSL','CONSPL','MZMSL','DTCOLNVHFNM','DTCTHFNM','INVEST',
        # Rates & FX
        'FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA','COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM','TWEXMMTH','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx',
        # Prices
        'PPIFGS','PPIFCG','PPIITM','PPICRM','OILPRICEx','PPICMM','NAPMPRI','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUUR0000SAD','CUSR0000SAS','CPIULFSL','CUUR0000SA0L2','CUSR0000SA0L5','PCEPI','DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA',
        # Stock Market
        'S&P 500','S&P: indust','S&P div yield','S&P PE ratio'
    ]
    def label_group(name: str) -> str:
        if name in ['RPI','W875RX1','INDPRO','IPFPNSS','IPFINAL','IPCONGD','IPDCONGD','IPNCONGD','IPBUSEQ','IPMAT','IPDMAT','IPNMAT','IPMANSICS','IPB51222s','IPFUELS','NAPMPI','CUMFNS']:
            return 'Output & Income'
        if name in ['HWI','HWIURATIO','CLF16OV','CE16OV','UNRATE','UEMPMEAN','UEMPLT5','UEMP5TO14','UEMP15OV','UEMP15T26','UEMP27OV','CLAIMSx','PAYEMS','USGOOD','CES1021000001','USCONS','MANEMP','DMANEMP','NDMANEMP','SRVPRD','USTPU','USWTRADE','USTRADE','USFIRE','USGOVT','CES0600000007','AWOTMAN','AWHMAN','NAPMEI','CES0600000008','CES2000000008','CES3000000008']:
            return 'Labor Market'
        if name in ['HOUST','HOUSTNE','HOUSTMW','HOUSTS','HOUSTW','PERMIT','PERMITNE','PERMITMW','PERMITS','PERMITW']:
            return 'Consumption & Housing'
        if name in ['DPCERA3M086SBEA','CMRMTSPLx','RETAILx','NAPM','NAPMNOI','NAPMSDI','NAPMII','ACOGNO','AMDMNOx','ANDENOx','AMDMUOx','BUSINVx','ISRATIOx','UMCSENTx']:
            return 'Orders & Inventories'
        if name in ['M1SL','M2SL','M2REAL','AMBSL','TOTRESNS','NONBORRES','BUSLOANS','REALLN','NONREVSL','CONSPL','MZMSL','DTCOLNVHFNM','DTCTHFNM','INVEST']:
            return 'Money & Credit'
        if name in ['FEDFUNDS','CP3Mx','TB3MS','TB6MS','GS1','GS5','GS10','AAA','BAA','COMPAPFFx','TB3SMFFM','TB6SMFFM','T1YFFM','T5YFFM','T10YFFM','AAAFFM','BAAFFM','TWEXMMTH','EXSZUSx','EXJPUSx','EXUSUKx','EXCAUSx']:
            return 'Rates & FX'
        if name in ['PPIFGS','PPIFCG','PPIITM','PPICRM','OILPRICEx','PPICMM','NAPMPRI','CPIAUCSL','CPIAPPSL','CPITRNSL','CPIMEDSL','CUSR0000SAC','CUUR0000SAD','CUSR0000SAS','CPIULFSL','CUUR0000SA0L2','CUSR0000SA0L5','PCEPI','DDURRG3M086SBEA','DNDGRG3M086SBEA','DSERRG3M086SBEA']:
            return 'Prices'
        if name in ['S&P 500','S&P: indust','S&P div yield','S&P PE ratio']:
            return 'Stock Market'
        return 'Other'
    series_to_group = {s: label_group(s) for s in ordered_series}
    return ordered_series, series_to_group
"""
Utilities for attaching group labels to DataFrames as MultiIndex columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional
import warnings

def get_fredmd_grouping():
    """
    Return (ordered_series, series_to_group) for the FRED-MD grouping 
    defined in the McCracken & Ng (2015) updated appendix.

    Update by feeding updated appendix from FRED to LLM, and then hardcoding the resulting grouping here. This is a bit
    brittle but ensures that the grouping is exactly as defined in the appendix.

    In addition, some series must be manually reviewed 
    (e.g., 02.17.2026 IPB51222s is defined with capital S in the data download, but lowercase s in the grouping).
    """
    ordered_series = [
        # Group 1: Output and Income
        'RPI', 'W875RX1', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 
        'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS', 'CUMFNS',
        
        # Group 2: Labor Market
        'HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 
        'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 'USGOOD', 'CES1021000001', 
        'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 
        'USFIRE', 'USGOVT', 'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008', 
        'CES2000000008', 'CES3000000008',
        
        # Group 3: Housing
        'HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW', 
        'PERMITS', 'PERMITW',
        
        # Group 4: Consumption, Orders, and Inventories
        'DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'ACOGNO', 'AMDMNOx', 'ANDENOx', 'AMDMUOx', 
        'BUSINVx', 'ISRATIOx', 'UMCSENTx',
        
        # Group 5: Money and Credit
        'M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 'REALLN', 
        'NONREVSL', 'CONSPI', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST',
        
        # Group 6: Interest and Exchange Rates
        'FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 
        'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 
        'BAAFFM', 'TWEXAFEGSMTHx', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx', 'EXCAUSx',
        
        # Group 7: Prices
        'WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM', 'CPIAUCSL', 
        'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 
        'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 
        'DNDGRG3M086SBEA', 'DSERRG3M086SBEA',
        
        # Group 8: Stock Market
        'S&P 500', 'S&P div yield', 'S&P PE ratio', 'VIXCLSx'
    ]

    def label_group(name: str) -> str:
        # Group 1
        if name in ['RPI', 'W875RX1', 'INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 
                    'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 
                    'IPB51222S', 'IPFUELS', 'CUMFNS']:
            return 'Output and Income'
        
        # Group 2
        if name in ['HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV', 'UNRATE', 'UEMPMEAN', 'UEMPLT5', 
                    'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 
                    'USGOOD', 'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP', 
                    'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT', 
                    'CES0600000007', 'AWOTMAN', 'AWHMAN', 'CES0600000008', 'CES2000000008', 
                    'CES3000000008']:
            return 'Labor Market'
        
        # Group 3
        if name in ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 
                    'PERMITMW', 'PERMITS', 'PERMITW']:
            return 'Housing'
        
        # Group 4
        if name in ['DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx', 'ACOGNO', 'AMDMNOx', 'ANDENOx', 
                    'AMDMUOx', 'BUSINVx', 'ISRATIOx', 'UMCSENTx']:
            return 'Consumption, Orders, and Inventories'
        
        # Group 5
        if name in ['M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 
                    'REALLN', 'NONREVSL', 'CONSPI', 'DTCOLNVHFNM', 'DTCTHFNM', 'INVEST']:
            return 'Money and Credit'
        
        # Group 6
        if name in ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 
                    'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 
                    'AAAFFM', 'BAAFFM', 'TWEXAFEGSMTHx', 'EXSZUSx', 'EXJPUSx', 'EXUSUKx', 
                    'EXCAUSx']:
            return 'Interest and Exchange Rates'
        
        # Group 7
        if name in ['WPSFD49207', 'WPSFD49502', 'WPSID61', 'WPSID62', 'OILPRICEx', 'PPICMM', 
                    'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 
                    'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 
                    'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA']:
            return 'Prices'
        
        # Group 8
        if name in ['S&P 500', 'S&P div yield', 'S&P PE ratio', 'VIXCLSx']:
            return 'Stock Market'
            
        return 'Other'

    series_to_group = {s: label_group(s) for s in ordered_series}
    return ordered_series, series_to_group


def add_group_level(df: pd.DataFrame,
                    series_to_group: Dict[str, str],
                    default_group: str = 'Other',
                    level_name: str = 'group',
                    strict: bool = True) -> pd.DataFrame:
    """
    Add a group level to a DataFrame's column Index (or MultiIndex).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    series_to_group : dict
        Mapping from series name to group label.
    default_group : str
        Label for series not found in the mapping (only used if strict=False).
    level_name : str
        Name for the new level.
    strict : bool
        If True, raise an error when columns are not found in the mapping.
        If False, assign them to `default_group` with a warning.

    Returns
    -------
    pd.DataFrame
        DataFrame with the extra group level in its column MultiIndex.

    Raises
    ------
    ValueError
        If strict=True and any column has no group assignment.
    """
    cols = df.columns

    if isinstance(cols, pd.MultiIndex):
        series_names = cols.get_level_values(-1)
    else:
        series_names = cols

    # --- Check for unmapped columns ---
    unmapped = [s for s in series_names if s not in series_to_group]
    if unmapped:
        unique_unmapped = sorted(set(unmapped))
        msg = (f"{len(unique_unmapped)} column(s) have no group assignment: "
               f"{unique_unmapped}")
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg + f" — assigning to '{default_group}'.")

    # --- Check for mapping entries that don't appear in the data ---
    unused = [s for s in series_to_group if s not in set(series_names)]
    if unused:
        warnings.warn(
            f"{len(unused)} entries in series_to_group are not present "
            f"in the DataFrame columns: {sorted(unused)}"
        )

    # --- Build the new MultiIndex ---
    groups = [series_to_group.get(s, default_group) for s in series_names]

    if isinstance(cols, pd.MultiIndex):
        new_tuples = []
        for tup, g in zip(cols, groups):
            new_tuples.append(tup[:1] + (g,) + tup[1:])
        names = [cols.names[0], level_name] + list(cols.names[1:])
        new_index = pd.MultiIndex.from_tuples(new_tuples, names=names)
    else:
        new_index = pd.MultiIndex.from_arrays(
            [groups, cols],
            names=[level_name, cols.name or 'series']
        )

    out = df.copy()
    out.columns = new_index
    return out


def build_full_group_mapping(fred_md: pd.DataFrame,
                             forward: pd.DataFrame,
                             yields: pd.DataFrame) -> Dict[str, str]:
    """
    Build a unified series_to_group mapping covering FRED-MD, forward rates,
    and yield columns.

    Raises
    ------
    ValueError
        If any FRED-MD columns are missing from the canonical grouping.
    """
    _, s2g = get_fredmd_grouping()

    # Check that every fred_md column has a group assigned
    missing_from_grouping = [c for c in fred_md.columns if c not in s2g]
    if missing_from_grouping:
        raise ValueError(
            f"The following FRED-MD columns have no group defined in "
            f"get_fredmd_grouping(): {missing_from_grouping}. "
            f"Please update the grouping function."
        )

    # Check that every entry in the grouping actually exists in fred_md
    missing_from_data = [c for c in s2g if c not in set(fred_md.columns)]
    if missing_from_data:
        warnings.warn(
            f"The following series are defined in get_fredmd_grouping() but "
            f"are not present in the FRED-MD data: {missing_from_data}. "
            f"They may have been dropped or renamed."
        )

    # Forward rates all belong to one group
    for c in forward.columns:
        s2g[c] = 'Forward Rates'

    # Yields all belong to one group
    for c in yields.columns:
        s2g[c] = 'Yields'

    return s2g


def get_group_indices(X: pd.DataFrame,
                      level: str = 'group') -> Dict[str, list]:
    """
    Return {group_name: [positional indices]} from a MultiIndex DataFrame.
    """
    if not isinstance(X.columns, pd.MultiIndex):
        raise ValueError("X must have a MultiIndex on columns")

    groups_array = X.columns.get_level_values(level)
    result = {}
    for i, g in enumerate(groups_array):
        result.setdefault(g, []).append(i)
    return result


def groups_as_array(X: pd.DataFrame, level: str = 'group'):
    """
    Return a numpy array of integer group codes, one per column.
    """
    if not isinstance(X.columns, pd.MultiIndex):
        raise ValueError("X must have a MultiIndex on columns")

    labels = X.columns.get_level_values(level)
    unique = list(dict.fromkeys(labels))  # preserves order
    mapping = {name: i for i, name in enumerate(unique)}
    return np.array([mapping[l] for l in labels])
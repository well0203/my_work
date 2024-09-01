import pandas as pd
import statsmodels.api as sm
from pmdarima.arima.utils import ndiffs, nsdiffs


# PP test is not needed to avoid redundancy and slow computation.
def stationary_seasonal(data: pd.DataFrame,
                        column_name: str) -> str:
    """
    Function that checks stationarity of a time series and 
    whether seasonal differencing is required or not. 
    Performs ADF and KPSS tests for trend and difference stationarity 
    checks, respectively. 
    Performs OCSB and CH tests for seasonal stationarity checks.

    Args:
        data (pd.DataFrame): Dataframe with time series data.
        column_name (str): The name of the column in the dataframe.
    
    Returns:
        column_name (str): The name of the column in the dataframe if seasonality
                            was found, None otherwise.
    """
    # https://alkaline-ml.com/pmdarima/1.3.0/tips_and_tricks.html?highlight=kpss
    # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    # https://de.mathworks.com/help/econ/trend-stationary-vs-difference-stationary.html

    # unit root, trend
    adf_diff = ndiffs(data[column_name], test='adf', max_d=90)

    # difference stationary
    kpss_diff = ndiffs(data[column_name], test='kpss', max_d=90)

    print(f'Column name: {column_name}')

    
    if adf_diff == 0:
        print('ADF: Is trend stationary.')
    else: 
        print('ADF:Is not trend stationary.')
        print('adf_diff', adf_diff)

    if kpss_diff == 0:
        print('KPSS: Is difference stationary.')
    else: 
        print('KPSS: Is not difference stationary.')
        print('kpss_diff', kpss_diff)

    # unit root
    ocsb_diff = nsdiffs(data[column_name], test='ocsb', m=24, max_D=5)
    
    # more sensitive, can find seasonality if there are complex or less pronounced patterns
    ch_diff = nsdiffs(data[column_name], test='ch', m=24, max_D=5)
    seasonal_diff = max(ocsb_diff, ch_diff)

    if seasonal_diff == 0:
        print('Does not require seasonal differencing')
    else:
        print('Requires seasonal differencing')
        print('ocsb_diff', ocsb_diff, 'ch_diff', ch_diff)

    print('-'*50)
    return column_name if seasonal_diff != 0 else None


def granger_causality(data: pd.DataFrame, 
                      col1: str, 
                      col2: str, 
                      test: str='ssr_ftest',
                      max_lag: int=12) -> None:
    """
    Performs Granger causality test for a time serie from col1 and col2.

    Args:
        data (pd.DataFrame): The input data as a pandas DataFrame.
        col1 (str): The name of the first time serie.
        col2 (str): The name of the second time serie.
        test (str): The type of Granger causality test to perform.
                    Possible alternatives are: 'ssr_ftest', 'ssr_chi2test', 'lrtest', 
                    'params_ftest' (default: 'ssr_ftest').
        max_lag (int): The maximum lag order to consider (default: 12).

    Returns:
        None
    """

    results = sm.tsa.stattools.grangercausalitytests(data[[col1, col2]], 
                                                     maxlag=max_lag, 
                                                     verbose=False)
    
    print(f"'{col1}' -> '{col2}':")
    
    
    important_lags = [lag for lag in range(1, max_lag+1) if results[lag][0][test][1] < 0.05]

    if len(important_lags) == 0:
        print("No important lags found.")
    else:
        print(f"Important lags up to lag {max_lag}: {important_lags} \n")
    
    return None
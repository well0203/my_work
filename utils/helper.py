import pandas as pd
from sklearn.preprocessing import StandardScaler
from pmdarima.arima.utils import ndiffs, nsdiffs


# I checked 2 libraries, they deliver same results. 
# This version is better, because outputs number of differences
# and bypass p-values interpretation. Better to check both,
# because pmdarima does not describe functions properly.

def stationary_seasonal(data, column_name):
    """
    Function that checks stationarity of a time series and 
    whether seasonal differencing is required or not.

    Args:
        data (pd.DataFrame): Dataframe with time series data.
        column_name (str): The name of the column in the dataframe.
    """
    # https://alkaline-ml.com/pmdarima/1.3.0/tips_and_tricks.html?highlight=kpss
    # https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html

    # unit root, trend
    adf_diff = ndiffs(data[column_name], test='adf', max_d=90)

    # difference stationary
    kpss_diff = ndiffs(data[column_name], test='kpss', max_d=90)

    print(f'Column name: {column_name}')

    
    if adf_diff == 0:
        print('ADF: Is stationary.')
    else: 
        print('ADF:Is not stationary.')
        print('adf_diff', adf_diff)

    if kpss_diff == 0:
        print('KPSS: Is stationary.')
    else: 
        print('KPSS: Is not stationary.')
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


def split_scale_dataset(data, train_size, val_size, test_size=None):

    """
    Function that splits and scales a time series dataset.

    Args:
        data (pd.DataFrame): Dataframe with time series data.
        train_size, test_size, val_size (int): number of days in train, 
                                               test and validation datasets.

    return: Scaled datasets
    """

    num_train = train_size*24
    if test_size is not None:
        num_test = test_size*24
    num_vali = val_size*24

    train_data = data.iloc[:num_train] # 0, a-1
    vali_data = data.iloc[num_train: num_train + num_vali] # a, a+b-1
    test_data = data.iloc[num_train + num_vali:] # a+b

    assert(len(data) == len(train_data) + len(test_data) + len(vali_data))

    print(f'{len(train_data)} observations in the train dataset.\n' 
          f'{len(test_data)} observations in the test dataset.\n' 
          f'{len(vali_data)} observations in the validation dataset.')

    # initialize scaler object
    scaler = StandardScaler()

    # scale data
    train_data_sc = scaler.fit_transform(train_data)
    vali_data_sc = scaler.transform(vali_data)
    test_data_sc = scaler.transform(test_data)

    train_data_sc = pd.DataFrame(train_data_sc, columns=train_data.columns, index=train_data.index)
    vali_data_sc = pd.DataFrame(vali_data_sc, columns=vali_data.columns, index=vali_data.index)
    test_data_sc = pd.DataFrame(test_data_sc, columns=test_data.columns, index=test_data.index)

    return train_data_sc, vali_data_sc, test_data_sc


def add_exog_vars(data, train_size, val_size, test_size=None):
    """
    Function that adds exogenous variables to a time series dataset.

    Args:
        data (pd.DataFrame): Dataframe with time series data.
        train_size, test_size, val_size (int): number of days in train, 
                                               test and validation datasets.

    return: Datasets with exogenous variables
    """
    num_train = train_size*24
    if test_size is not None:
        num_test = test_size*24
    num_vali = val_size*24

    train_data = data.iloc[:num_train][['HourOfDay', 'DayOfWeek']] # 0, a-1
    vali_data = data.iloc[num_train: num_train + num_vali][['HourOfDay', 'DayOfWeek']] # a, a+b-1
    test_data = data.iloc[num_train + num_vali:][['HourOfDay', 'DayOfWeek']] # a+b

    return train_data, vali_data, test_data


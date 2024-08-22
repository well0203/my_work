import pandas as pd
from sklearn.preprocessing import StandardScaler
from pmdarima.arima.utils import ndiffs, nsdiffs


def stationary_seasonal(data, column_name):
    # https://alkaline-ml.com/pmdarima/1.3.0/tips_and_tricks.html?highlight=kpss
    
    adf_diff = ndiffs(data[column_name], test='adf')
    kpss_diff = ndiffs(data[column_name], test='kpss')
    pp_diff = ndiffs(data[column_name], test='pp')
    
    if adf_diff == 0 and kpss_diff == 0 and pp_diff == 0:
         print('Is stationary')
    else: 
        print('Is not stationary')

    if adf_diff > 0 or kpss_diff > 0:
        ocsb_diff = nsdiffs(data[column_name], test='ocsb', m=24)
        ch_diff = nsdiffs(data[column_name], test='ch', m=24)
        if ocsb_diff == 0 and ch_diff==0:
            print('Does not require seasonal differencing')
        else:
            print('Requires seasonal differencing')

def split_scale_dataset(data, train_size, val_size, test_size=None):

    """
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
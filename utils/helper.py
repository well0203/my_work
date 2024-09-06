import pandas as pd
import re
from sklearn.preprocessing import StandardScaler


def split_scale_dataset(data: pd.DataFrame, 
                        train_split: float, 
                        test_split: float
                        ) -> tuple[pd.DataFrame, 
                                   pd.DataFrame, 
                                   pd.DataFrame]:
    """
    Function that splits and scales a time series dataset.

    Args:
        data (pd.DataFrame): Dataframe with time series data.
        train_split, test_split (float): Proportion of data in train and 
                                         test datasets.

    Returns: 
        train_data_sc, vali_data_sc, test_data_sc (pd.DataFrame): Scaled datasets
    """

    # number of days 
    train_size = int(round(len(data)/24*train_split, 0))
    test_size = int(round(len(data)/24*(1-train_split-test_split), 0))

    # calculate number of observations in each dataset
    num_train = train_size*24
    num_test = test_size*24
    num_vali = len(data) - num_train - num_test 

    # split data into datasets
    train_data = data.iloc[:num_train] # 0, a-1
    vali_data = data.iloc[num_train: num_train + num_vali] # a, a+b-1
    test_data = data.iloc[num_train + num_vali:] # a+b

    # check that the data is split correctly
    assert(len(data) == len(train_data) + len(test_data) + len(vali_data))

    # print number of observations in each dataset
    print(f'{len(train_data)} observations in the train dataset.\n' 
          f'{len(vali_data)} observations in the validation dataset. \n'
          f'{len(test_data)} observations in the test dataset.')

    # initialize scaler object
    scaler = StandardScaler()

    # scale data
    train_data_sc = scaler.fit_transform(train_data)
    vali_data_sc = scaler.transform(vali_data)
    test_data_sc = scaler.transform(test_data)

    # convert scaled data back to pandas DataFrames
    train_data_sc = pd.DataFrame(train_data_sc, columns=train_data.columns, index=train_data.index)
    vali_data_sc = pd.DataFrame(vali_data_sc, columns=vali_data.columns, index=vali_data.index)
    test_data_sc = pd.DataFrame(test_data_sc, columns=test_data.columns, index=test_data.index)

    return train_data_sc, vali_data_sc, test_data_sc


def add_exog_vars(train_data: pd.DataFrame,
                  vali_data: pd.DataFrame,
                  test_data: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function that adds exogenous variables to a time series datasets.

    Args:
        train_data, vali_data, test_data (pd.DataFrame): Dataframe with time series data.
       
    Returns: 
    pd.DataFrame *3: Datasets with exogenous variables
    """
    
    # add exogenous variables
    for dataset in [train_data, vali_data, test_data]:
        dataset['DayOfWeek'] = dataset.index.dayofweek
        dataset['HourOfDay'] = dataset.index.hour

    return train_data, vali_data, test_data


# Function to find and extract metrics from command output
def extract_metrics_from_output(output):
    mse = None
    mae = None

    # Regex patterns to search for MSE and MAE
    mse_pattern = re.compile(r"mse:\s*([\d.]+)", re.IGNORECASE)
    mae_pattern = re.compile(r"mae:\s*([\d.]+)", re.IGNORECASE)

    # Iterate each line to find values
    for line in output:
        if mse is None:  # Find MSE if not already found
            mse_match = mse_pattern.search(line)
            if mse_match:
                mse = float(mse_match.group(1))
        if mae is None:  # Find MAE if not already found
            mae_match = mae_pattern.search(line)
            if mae_match:
                mae = float(mae_match.group(1))
        if mse is not None and mae is not None:
            break  # Stop if both metrics are found

    return mse, mae
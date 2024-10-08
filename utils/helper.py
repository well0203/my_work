import os
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    

def split_scale_dataset(data: pd.DataFrame, 
                        train_split: float, 
                        test_split: float,
                        scaler_type: str = 'standard',
                        inverse_transform: bool = False
                        ) -> tuple[pd.DataFrame, 
                                   pd.DataFrame, 
                                   pd.DataFrame]:
    """
    Function that splits and scales a time series dataset.

    Args:
        data (pd.DataFrame): Dataframe with time series data.
        train_split, test_split (float): Proportion of data in train and 
                                         test datasets.
        scaler_type (str): Scaler to be used. Possible values: 'standard' 
                           or 'minmax', or 'minmax2'. The latter uses a range 
                           of 0 to 5 (default: 'standard').
        inverse_transform (bool): If True, inverse transform the data (default: False).

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
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'minmax2':
        scaler = MinMaxScaler(feature_range=(0, 5))
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError('Invalid scaler type')

    # scale data
    train_data_sc = scaler.fit_transform(train_data)
    vali_data_sc = scaler.transform(vali_data)
    test_data_sc = scaler.transform(test_data)

    if inverse_transform:
        train_data_sc = scaler.inverse_transform(train_data_sc)
        vali_data_sc = scaler.inverse_transform(vali_data_sc)
        test_data_sc = scaler.inverse_transform(test_data_sc)

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


def extract_metrics_from_output(output, 
                                itr=2,
                                if_scaled=True
                                ) -> list[tuple]:
    """
    Function to extract metrics from command output.
    
    Args:
        output (list): List of strings containing the command output.
        itr (int): Number of iterations to extract metrics for (default: 2).
        if_scaled (bool): Whether the output should be scaled (default: True).
        
    Returns:
        list[tuple]: List of tuples containing the extracted metrics 
                     for each iteration converted to floats.
    """

    # Pattern for extracting metrics
    if if_scaled:
        pattern = re.compile(
            r"Scaled mse:\s*([\d.]+),\s*rmse:\s*([\d.]+),\s*mae:\s*([\d.]+),\s*rse:\s*([\d.]+)",
            re.IGNORECASE
        )
    else:
        pattern = re.compile(
            r"Original data scale mse:\s*([\d.]+),\s*rmse:\s*([\d.]+),\s*mae:\s*([\d.]+),\s*rse:\s*([\d.]+)",
            re.IGNORECASE
        )
    # Join the output lines to a single string
    output_str = " ".join(output)
    
    # Find all matches of the pattern
    matches = pattern.findall(output_str)
    
    # Throw an error if there are not enough matches
    if len(matches) < itr:
        raise ValueError(f"Expected at least {itr} iterations, but found only {len(matches)}.")
    
    # List with tuples of metrics for all iterations
    # Map string matches to floats
    return [tuple(map(float, match)) for match in matches[:itr]]


def convert_results_into_df(results, 
                            path_dir,
                            csv_name
                            ) -> pd.DataFrame:
    """
    Function to convert results into a pandas DataFrame.
    
    Args:
        results (list): List of tuples containing the results.
        path_dir (str): Path to the directory where the results will be saved.
        csv_name (str): Name of the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Set multi-index 
    df.set_index(['Loss_function', 'Iteration', 'Pred_len'], inplace=True)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(path_dir, csv_name), index=True)

    return df
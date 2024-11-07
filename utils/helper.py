import os
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def running_time(start_time: float, 
                 end_time: float
                 ) -> tuple[int, int, float]:
    """
    Function that returns hours, minutes and seconds of running time 
    of a function.

    Args:
        start_time (float): Start time from package "time".
        end_time (float): End time from package "time".

    Returns:
         Running time in hours, minutes and seconds.
    """
    hours, rem = divmod(end_time - start_time, 3600)
    mins, secs = divmod(rem, 60)

    return int(hours), int(mins), secs
    

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
        inverse_transform (bool): If True, inverse transform the data 
                                  (default: False).

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
    train_data_sc = pd.DataFrame(train_data_sc, 
                                 columns=train_data.columns, 
                                 index=train_data.index
                                 )
    vali_data_sc = pd.DataFrame(vali_data_sc, 
                                columns=vali_data.columns, 
                                index=vali_data.index
                                )
    test_data_sc = pd.DataFrame(test_data_sc, 
                                columns=test_data.columns, 
                                index=test_data.index
                                )

    return train_data_sc, vali_data_sc, test_data_sc


def add_exog_vars(train_data: pd.DataFrame,
                  vali_data: pd.DataFrame,
                  test_data: pd.DataFrame
                   ) -> tuple[pd.DataFrame, 
                              pd.DataFrame, 
                              pd.DataFrame]:
    """
    Function that adds exogenous variables to a time series datasets.

    Args:
        train_data, vali_data, test_data (pd.DataFrame): Dataframe with 
                                                         time series data.
       
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
                                if_scaled=True,
                                if_supervised=True
                                ) -> list[tuple]:
    """
    Function to extract metrics from command output.
    
    Args:
        output (list): List of strings containing the command output.
        itr (int): Number of iterations to extract metrics for (default: 2).
        if_scaled (bool): Whether the output should be scaled (default: True).
        if_supervised (bool): Whether the model is supervised (default: True).
        
    Returns:
        list[tuple]: List of tuples containing the extracted metrics 
                     for each iteration converted to floats.
    """
    # Join the output lines to a single string
    output_str = " ".join(output)

    if if_supervised:
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
        
        # Find all matches of the pattern
        matches = pattern.findall(output_str)
        
        # Throw an error if there are not enough matches
        if len(matches) < itr:
            raise ValueError(f"Expected at least {itr} iterations, but found only {len(matches)}.")
        
        # List with tuples of metrics for all iterations
        # Map string matches to floats
        return [tuple(map(float, match)) for match in matches[:itr]]

    else:
        # Pattern for extracting metrics
        mse_regex = re.compile(r"\bmse:\s*([\d\.]+)")
        rmse_regex = re.compile(r"\brmse:\s*([\d\.]+)")
        mae_regex = re.compile(r"\bmae:\s*([\d\.]+)")

        mse_match = mse_regex.search(output_str)
        rmse_match = rmse_regex.search(output_str)
        mae_match = mae_regex.search(output_str)

        if mse_match and rmse_match and mae_match:
            mse = float(mse_match.group(1))
            rmse = float(rmse_match.group(1))
            mae = float(mae_match.group(1))

            return [(mse, rmse, mae)]

        else:
            raise ValueError("Could not extract metrics from command output.")


def convert_results_into_df(results, 
                            path_dir = None,
                            csv_name = None,
                            if_loss_fnc=True,
                            itr=2
                            ) -> pd.DataFrame:
    """
    Function to convert results into a pandas DataFrame.
    
    Args:
        results (list): List of tuples/dictionaries containing 
                        the results.
        path_dir (str): Path to the directory where the results 
                        will be saved (default: None).
        csv_name (str): Name of the CSV file to save the results 
                        (default: None).
        if_loss_fnc (bool): Whether the loss function is included 
                            in the results (default: True).
        itr (int): Number of iterations to extract metrics for 
                   (default: 2).
        
    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    if if_loss_fnc:
        # Set multi-index 
        df.set_index(['Loss_function', 'Iteration', 'Pred_len'], inplace=True)

        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(path_dir, csv_name), index=True)

    else:
        if itr > 1:
            df.set_index(['Country', 'Pred_len', 'Iteration'], inplace=True)
            df = df.groupby(['Country', 'Pred_len']).mean()
        else:
            df.set_index(['Country', 'Pred_len'], inplace=True)

    return df


def highlight_value(serie,
                         fnc='min'
                  ) -> pd.Series:
    """
    Function to highlight the minimum or maximum value in bold.

    Args:
        serie (pd.Series): The input series.
        fnc (str): The function to apply. Possible values: 'min' or 'max'.

    Returns:
        pd.Series: The series with the minimum value highlighted in bold.
    """

    if fnc == 'min':
        is_min = serie == serie.min()
    elif fnc == 'max':
        is_min = serie == serie.max()
    else:
        raise ValueError("'fnc' must be 'min' or 'max'.")

    return ['font-weight: bold' if value else '' for value in is_min]  


def highlight_second_value(series: pd.Series, 
                           fnc: str = 'min'
                           ) -> list:
    """
    Function to highlight the second lowest or second highest value in bold,
    based on the provided function ('min' or 'max').

    Args:
        series (pd.Series): The input series.
        fnc (str): Function to determine whether to highlight the second 
                   lowest ('min') or second highest ('max') value.

    Returns:
        list: The list of styles.
    """

    if fnc == 'min':
        sorted_values = series.nsmallest(2)
    elif fnc == 'max':
        sorted_values = series.nlargest(2)
    else:
        raise ValueError("fnc argument must be 'min' or 'max'")
    
    # Check if there are at least two values
    if len(sorted_values) == 2:
        second_extreme = sorted_values.iloc[-1]  
        return [
            'text-decoration: underline;' if v == second_extreme else '' 
            for v in series
        ]
    
    return [''] * len(series)



def style_dataframe(df: pd.DataFrame,
                    fnc: str='min',
                    decimal_places=4,
                    second_value=False
             ) -> pd.DataFrame:
    """
    Function to style the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        fnc (str): The function to apply. Possible values: 'min' or 'max'.
        decimal_places (int): The number of decimal places to round the 
                              values to.
        second_value (bool): Whether to highlight the second lowest or 
                             second highest value underlined, based on the 
                             provided function ('min' or 'max').
                             Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with the style applied.
    """

    if fnc not in ['min', 'max']:
        raise ValueError("'fnc' must be 'min' or 'max'.")

    format_string = f"{{:.{decimal_places}f}}"
    
    # Apply only to 'RMSE' columns
    styled_df = df.style.format(format_string).apply(
        lambda x: highlight_value(x, fnc=fnc), 
        subset=pd.IndexSlice[:, (slice(None), 'RMSE')], 
        axis=1 
    )
    
    # Apply only to 'MAE' columns
    styled_df = styled_df.format(format_string).apply(
        lambda x: highlight_value(x, fnc=fnc), 
        subset=pd.IndexSlice[:, (slice(None), 'MAE')], 
        axis=1 
        )

    if second_value:
        styled_df = styled_df.apply(
            lambda x: highlight_second_value(x, fnc), 
            subset=pd.IndexSlice[:, (slice(None), 'MAE')], 
            axis=1 
        )
        styled_df = styled_df.apply(
            lambda x: highlight_second_value(x, fnc), 
            subset=pd.IndexSlice[:, (slice(None), 'RMSE')], 
            axis=1 
        )

    return styled_df


def map_country_name(country_code):
    """
    Function to map country names to country codes
    Args:
        country_code (str): Country code

    Returns:
        str: Country name
    """
    mapping_dict = {
        'DE': 'Germany',
        'GB': 'United Kingdom',
        'ES': 'Spain',
        'FR': 'France',
        'IT': 'Italy'
    }

    return mapping_dict[country_code]


def read_results_csv(file_path, 
                    columns_to_extract = ('-RevIN', ['RMSE', 'MAE'])):
    """
    Read data from a csv file from our multiindex results.
    Args:
        file_path (str): CSV file name with its corresponding folder.
        columns_to_extract (tuple): The columns to extract from the CSV file.
                                    Should be in the format (model, ['RMSE', 'MAE']).
                                    Defaults to ('-RevIN', ['RMSE', 'MAE']).
    Returns:
        pd.DataFrame: The resulted DataFrame from the CSV file.
    """

    path_dir = 'results/'
    path = os.path.join(path_dir, file_path)

    return pd.read_csv(path, 
                       header=[0, 1], 
                       index_col=[0, 1]
                       ).loc[:, columns_to_extract]


def group_and_reindex_df(data,
                         to_group='Country'
                         ) -> pd.DataFrame:
    """
    Groups the dataframe by a specific index and reindexes 
    the result to match the original order.

    Args:
        data (pandas.DataFrame): The dataframe to group and reindex.
        to_group (str): The index to group by and reindex (default: 'Country')
                        can be 'Pred_len' or both, but then needs to be a list.
    """
    if type(to_group) != list:
        original_order = data.index.get_level_values(to_group).unique()
        data = data.groupby(level=to_group).mean()
        data = data.reindex(original_order)
    else:
        data = data.mean(axis=0).reset_index().rename(columns={0:'Value'})
    
    return data


def select_model(country: str, 
                 pred_len: int):
    """
    Select the best model for the given country and prediction length.

    Args:
    country (str): The name of the country.
    pred_len (int): The prediction length.

    Returns
    (str): The name of the best model for the given country and prediction length.

    """
    if country == 'Germany' and pred_len == 24:
        return 'PatchTST/42'
    elif country in ['Germany', 'United Kingdom']:
        return 'PatchTST/64'
    elif country == 'Spain':
        return 'PatchTST/42'
    elif country in ['France', 'Italy']:
        return 'PatchTST/21'


def choose_best_patchtst_model(data: pd.DataFrame
                               ) -> pd.DataFrame:
    """
    Choose best model for the given country and prediction length
    using select_model() function. Creates a DataFrame with the best model 
    for each country and prediction length.

    Args:
    data (pd.DataFrame): The DataFrame containing the results.

    Returns
    result_df (pd.DataFrame): The DataFrame containing the best model 
                              for each country and prediction length.
    """

    selected_dfs = []
    for (country, pred_len) in data.index:
        model = select_model(country, pred_len)
        selected_df = data.loc[[(country, pred_len)], pd.IndexSlice[model, :]]  # Double brackets to get a DataFrame.
        selected_df.columns = pd.MultiIndex.from_tuples(
            [('PatchTST', metric) for metric in selected_df.columns.get_level_values('Metrics')],
            names=['Model', 'Metrics']
        )
        selected_dfs.append(selected_df)

    result_df = pd.concat(selected_dfs)

    return result_df


def second_best_value(series: pd.Series
                      ) -> float:
    """Find the second-best value in the series.

    Args:
        series (pd.Series): The input series.

    Returns:
        float: The second-best value in the series.
    """

    sorted_values = series.sort_values(ascending=True)
    second_best = sorted_values.iloc[1]  
    return second_best


def calculate_improvement(data: pd.DataFrame, 
                          base_mae_model: str, 
                          model_to_compare_mae: str = None, 
                          base_rmse_model: str = None, 
                          model_to_compare_rmse:str = None, 
                          if_return_values: bool = False,
                          grouped_by_models: bool = True
                          )-> tuple[float, float] | None | pd.DataFrame:
    """
    Calculate the improvement of a base model over a second-best model 
    in terms of MAE and RMSE.

    Args:
        data (DataFrame): DataFrame containing model performance with 
                          'Model', 'Metrics', and 'Value' columns.
        base_mae_model (str): The name of the base model for MAE improvement 
                              comparison.
        base_rmse_model (str, optional): The name of the base model for RMSE 
                                         improvement comparison. Defaults to the 
                                               same as base_mae_model.
        model_to_compare_mae (str): The name of the model to compare against for 
                                    MAE. If None, compare against the second-best model.
                                    Defaults to None.
        model_to_compare_rmse (str, optional): The name of the model to compare 
                                               against for RMSE. Defaults to the 
                                               same as model_to_compare_mae.
        if_return_values (bool, optional): Whether to return the improvement 
                                           percentages for MAE and RMSE. 
                                           Defaults to False.
        grouped_by_models (bool, optional): Whether to calculate overall improvement 
                                            by models. Defaults to True.

    Returns:
        improvement_percentage_mae, 
        improvement_percentage_rmse 
        (tuple[float, float]) | None | pd.DataFrame: A tuple containing the improvement 
                                                     percentages for MAE and RMSE, or None,
                                                     if grouped_by_models is True.
                                                     DataFrame containing the improvement 
                                                     percentages for MAE and RMSE, if 
                                                     grouped_by_models is False.
    """
    if model_to_compare_rmse is None:
        model_to_compare_rmse = model_to_compare_mae
    if base_rmse_model is None:
        base_rmse_model = base_mae_model

    if grouped_by_models:

        # Calculate MAE improvement
        base_mae = data.loc[
            (data['Model'] == base_mae_model) & (data['Metrics'] == 'MAE'), 'Value'
        ].values

        compare_mae = data.loc[
            (data['Model'] == model_to_compare_mae) & (data['Metrics'] == 'MAE'), 'Value'
        ].values

        improvement_percentage_mae = ((compare_mae - base_mae) / 
                                    compare_mae * 100)[0].round(2)
        print(f'Improvement of {base_mae_model} over {model_to_compare_mae} ' 
            f'in terms of MAE: {improvement_percentage_mae} %')

        # Calculate RMSE improvement
        base_rmse = data.loc[
            (data['Model'] == base_rmse_model) & (data['Metrics'] == 'RMSE'), 'Value'
        ].values

        compare_rmse = data.loc[
            (data['Model'] == model_to_compare_rmse) & (data['Metrics'] == 'RMSE'), 'Value'
        ].values

        improvement_percentage_rmse = ((compare_rmse - base_rmse) / 
                                    compare_rmse * 100)[0].round(2)
        print(f'Improvement of {base_rmse_model} over {model_to_compare_rmse} ' 
            f'in terms of RMSE: {improvement_percentage_rmse} %')
        if if_return_values:
            return improvement_percentage_mae, improvement_percentage_rmse

    else:
        if model_to_compare_mae is None:
            
            first_best_rmse = data.xs(base_mae_model, axis=1, level=0)['RMSE']
            second_best_rmse = data.xs('RMSE', axis=1, level=1).apply(lambda x: second_best_value(x), axis=1)

            data['improvement_rmse'] = ((second_best_rmse - first_best_rmse) / 
                                    second_best_rmse * 100).round(2)

            first_best_mae = data.xs(base_mae_model, axis=1, level=0)['MAE']
            second_best_mae = data.xs('MAE', axis=1, level=1).apply(lambda x: second_best_value(x), axis=1)

            data['improvement_mae'] = ((second_best_mae - first_best_mae) / 
                                    second_best_mae * 100).round(2)
            
        else:
            data['rmse_improvement'] = ((
                (data.loc[:, (model_to_compare_rmse, 'RMSE')] - \
                data.loc[:, (base_rmse_model, 'RMSE')]) / \
                data.loc[:, (model_to_compare_rmse, 'RMSE')]
                ) * 100).round(2)
            data['mae_improvement'] = ((
                (data.loc[:, (model_to_compare_mae, 'MAE')] - \
                data.loc[:, (base_mae_model, 'MAE')]) / \
                data.loc[:, (model_to_compare_mae, 'MAE')]
                ) * 100).round(2)
        
        return data

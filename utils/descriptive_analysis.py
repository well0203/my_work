import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def plot_missings(data: pd.DataFrame, 
                  rotation: int = 15
                  ):
    """
    Plots missing values.

    Args:
        data (pandas.DataFrame): The time series to plot.
        rotation (int): The rotation of the x-axis labels (default: 15).

    Returns:
        None
    """

    plt.figure(figsize=(25, 15))
    sns.heatmap(data.isnull(), cbar=False, cmap='YlGnBu')
    plt.title('Missing values over time', fontsize=25)
    plt.xlabel('Columns')
    plt.ylabel('Timestamp')
    plt.xticks(rotation=rotation, fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_seasonality(data: pd.DataFrame, 
                     frequency: str = 'Month', 
                     hue: str = 'Year', 
                     num_cols: int = 3, 
                     title: str = None
                     ):
    """
    Plots the seasonality of the time series.

    Args:
        data (pandas.DataFrame): The time series to plot.
        frequency (str): The frequency of the time series (default: 'Month').
        hue (str): Subsets of the data (default: 'Year').
        num_cols (int): Number of plots
        title (str): The title of the plot (default: None).
    Returns:
        None
    """

    num_cols = num_cols
    num_rows = 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

    for i, ax in enumerate(axes.flat):
        if i < len(data.columns):
            col = data.columns[i]
            sns.lineplot(ax=ax, data=data, x=frequency, y=col, 
                         hue=hue, palette='viridis', errorbar=None,
                         linewidth=2.5, markers=True)
            ax.set_title(f'{col}')
            ax.set_xlabel(frequency)
            ax.set_ylabel('Value')
            ax.legend(title=hue, loc = 'upper right')
            
        else:
            ax.axis('off')  

    fig.suptitle(title, fontsize=25)   
    plt.tight_layout()
    plt.show()


def hist_plots(data: pd.DataFrame, 
               col: str, 
               num_cols: int=3, 
               title: str=None
               ):

    """
    Plots the frequency of values in time series.

    Args:
        data (pandas.DataFrame): The time series to plot.
        col (str): The country name to plot.
        num_cols (int): Number of plots per country.
        title (str): The title of the plot (default: None).

    Returns:
        None
    """

    num_cols = num_cols
    num_rows = 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

    for i, ax in enumerate(axes.flat):
        if i < len(data.columns):
            col = data.columns[i]
            sns.histplot(ax=ax, data=data, x=col)
            ax.set_title(f'{col}')
        else:
            ax.axis('off')  

    fig.suptitle(title, fontsize=25)   
    plt.tight_layout()
    plt.show()


def corr_plot(data: pd.DataFrame, 
              annot: bool = True, 
              mask: bool = True, 
              title: str = None
              ):
    """
    Plots the correlation between time series with customization options.

    Args:
        data (pandas.DataFrame): The time series to plot.
        annot (bool, optional): Show correlation values on the heatmap (default: True).
        mask (bool, optional): Mask the diagonal elements of the heatmap (default: True).
        title (str, optional): The title of the plot (default: None).

    Returns:
        None
    """
    
    data.columns = [change_col_name(col) for col in data.columns]
    # Compute correlation matrix
    corr_matrix = data.corr()

    if mask:
        mask_array = np.eye(len(corr_matrix), dtype=bool)
    else:
        mask_array = None

    plt.figure(figsize=(5, 3))  

    # Create the heatmap
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        mask=mask_array,
        cbar_kws={"shrink": 0.75},
        vmin=-1,
        vmax=1
        )

    plt.title(title, fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(fontsize=6) 
    plt.tight_layout() 
    plt.show()


def count_outliers(col: pd.Series, 
                   extreme: bool = False, 
                   verbose: bool = False, 
                   if_return: bool = False
                   ) -> tuple[float, float] | None:
    """
    Defines the number of outliers in a column using the IQR method.

    Args:
        col (pd.Series): Column with values to check.
        extreme (bool): Argument to check for extreme outliers 
                        (default: False).
        verbose (bool): Argument to print the upper, lower bounds,
                        as well as minimum and maximum values in a 
                        column (default: False).
        if_return (bool): Argument to return the bounds (upper, lower)
                          (default: False).

    Returns:
        upper, lower (float) or None
    """

    if extreme:
        factor = 3
    else:
        factor = 1.5

    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR

    upper = Q3 + factor * IQR

    outliers = (col < lower) | (col > upper)
    perc_outl = outliers.sum() / len(col) * 100

    if verbose:
        min_ = col.min()
        max_ = col.max()
        if max_ > upper or min_ < lower:
            print(f"Column name: '{col.name}'")
            print(f"lower bound: {lower:>15},   upper bound: {upper:>15}")
            print(f"min value:   {min_:>15},    max:         {max_:>15}")
            print('-' * 80)
    else:
        print(f"{col.name:<40} {outliers.sum():>15} {perc_outl:>20.1f}%")

    if if_return:
        return upper, lower
    else:
        return None
    

def periodograms(data: pd.DataFrame, 
                 num_cols: int = 3, 
                 max_ylim: int = 4e8, 
                 title: str = None
                 ):
    """
    Plots periodograms of multiple time series.

    Args:
        data (pandas.DataFrame): The time series to plot.
        num_cols (int): Number of columns in the subplot grid (default: 3).
        max_ylim (int): Maximum limit for the y-axis (default: 4e8).
        title (str): The title of the overall plot (default: None).

    Returns:
        None
    """
    
    # Create subplots
    fig, axes = plt.subplots(1, num_cols, figsize=(20, 5 * 1))

    sns.set_theme(style="whitegrid", palette="muted")
    
    for i, ax in enumerate(axes.flat):
        if i < len(data.columns):
            col = data.columns[i]

            # Code from TensorFlow Tutorial
            fft = tf.signal.rfft(data[col])
            f_per_dataset = np.arange(0, len(fft))
            
            n_samples_h = len(data)
            hours_per_year = 24 * 365.2524
            years_per_dataset = n_samples_h / hours_per_year
            f_per_year = f_per_dataset / years_per_dataset
            
            ax.plot(f_per_year, np.abs(fft), 
                    drawstyle='steps-mid', 
                    color=sns.color_palette("muted")[i % len(sns.color_palette("muted"))], 
                    label="Amplitude")
            
            ax.set_xscale('log')
            if np.abs(fft).max() > max_ylim:
                ax.set_ylim(0, max_ylim)
            ax.set_xlim([0.1, max(f_per_year)])
            ax.set_xticks([1, 365.2524])
            ax.set_xticklabels(['1/Year', '1/Day'])
            
            # Labels and title
            ax.set_xlabel('Frequency (log scale)', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.set_title(col, fontsize=14)        
            # Add grid and legend
            ax.grid(True, which='both', linestyle='--', linewidth=0.7)
            ax.legend()
        else:
            ax.axis('off')
    
    # Adjust layout and title
    if title is not None:
        fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def get_season(month: int
               ) -> str:
    """
    Function to map months to seasons
    Args:
        month (int): Month number

    Returns:
        str: Season
    """

    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'
    

def change_col_name(col_name: str
                    ) -> str:
    """
    Function to change column names to the desired format.
    
    Args:
        col_name (str): Original column name

    Returns:
        str: Desired column name
    """

    if 'GB' in col_name and 'load' in col_name:
        i = 2
        j = 3
    elif 'load' in col_name:
        i = 1
        j = 3
    elif 'GB' in col_name:
        i = 2
        j = 1
    else:
        i = 1
        j = 1
    
    col_name = ' '.join(col_name.split('_')[i:-j])
    return col_name


def stacked_bar_plot_per_season(data: pd.DataFrame, 
                                season_order: list = ['winter', 'spring', 'summer', 'autumn'], 
                                country: str = None):
    """
    Plots the average of values per season.

    Args:
        data (pandas.DataFrame): The raw time series data.
        season_order (list): The order of the seasons for the plot 
                            (default: ['winter', 'spring', 'summer',
                            'autumn']).
        country (str): The country used in the title of the plot 
                       (default: None).

    Returns:
        None
    """
    
    # Create a season column
    data['season'] = data.index.month.map(get_season)
    data.columns = [change_col_name(col) if col != 'season' else col for col in data.columns]

    # Group data by season and calculating the means 
    seasonal_avg = data.groupby('season'
                                ).agg('mean'
                                      ).reset_index(
                                          ).sort_values('season', 
                                                        key=lambda x: pd.Categorical(x, 
                                                                                     categories=season_order, 
                                                                                     ordered=True))

    melted_df = pd.melt(seasonal_avg, id_vars=['season'], 
                        value_vars=seasonal_avg.columns[1:])

    # Create the figure
    plt.figure(figsize=(4, 4))

    # Create a bar plot with two datasets
    sns.barplot(x='season', y='value', hue='variable', data=melted_df, palette="CMRmap")

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), prop={'size': 8})
    plt.xlabel('Season', fontsize=10)
    plt.ylabel('MW', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(f"Average Load, Solar Generation, and Wind Generation by Season for {country}", fontsize=10)
    plt.show()


def heatmap_plot(data: pd.DataFrame, 
                 country: str, 
                 x_axis: str = 'Month', 
                 y_axis: str = 'HourOfDay', 
                 cmap: list = ['Blues', 'Oranges', 'YlGnBu']):
    """
    Plots heatmaps for given time series data with flexible x and y axis groupings.

    Args:
        data (pandas.DataFrame): The DataFrame containing time series data.
        country (str): The name of the country to plot heatmaps for.
        x_axis (str): The x-axis grouping (default: 'Month').
        y_axis (str): The y-axis grouping (default: 'HourOfDay').
        cmap (list): A list of color maps for each heatmap (default: 
                    ['Blues', 'Oranges', 'YlGnBu']).

    Returns:
        None
    """

    result_col_names = [change_col_name(col) for col in data.columns]

    change_axis_name = {
        'Month': 'Month',
        'DayOfWeek': 'Day Of Week',
        'HourOfDay': 'Hour'
    }
    new_x_axis = change_axis_name[x_axis]
    new_y_axis = change_axis_name[y_axis]

    columns_to_plot = [col for col in data.columns if col not in [x_axis, y_axis]]

    # Pivot tables
    heatmaps = [
        data.pivot_table(index=y_axis, columns=x_axis, values=col, aggfunc='mean')
        for col in columns_to_plot
    ]
    
    fig, axs = plt.subplots(1, len(columns_to_plot), figsize=(5 * len(columns_to_plot), 5))

    # Plot each heatmap
    for i, (df, ax) in enumerate(zip(heatmaps, axs if len(columns_to_plot) > 1 else [axs])):
        sns.heatmap(df, cmap=cmap[i % len(cmap)], ax=ax, cbar_kws={'label': f'Avg {result_col_names[i]} (MW)'})
        ax.set_xlabel(new_x_axis, fontsize=12)
        ax.set_ylabel(new_y_axis, fontsize=12)
        
        # Customize tick labels
        if x_axis == 'Month':
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
        elif x_axis == 'DayOfWeek':
            ax.set_xticks(range(7))
            ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=30)
        
        if y_axis == 'DayOfWeek':
            ax.set_yticks(range(7))
            ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
        elif y_axis == 'Month':
            ax.set_yticks(range(1, 13))
            ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
        
        ax.set_title(f'{result_col_names[i]}', fontsize=14)

    plt.suptitle(f'Heatmaps for {country} by {new_y_axis} and {new_x_axis}', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_correlations_between_2vars(df: pd.DataFrame, 
                                    var1: str, 
                                    var2: str, 
                                    ax: plt.Axes, 
                                    color: str, 
                                    lags: range = range(1, 49)):
    """
    Plots correlation coefficients as a function of time lag.

    Args:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        var1 (str): The name of the first variable.
        var2 (str): The name of the second variable.
        lags (range): The range of lags tested (default: range(1, 49)).
        ax (matplotlib.axes.Axes): The axis object for the plot.
        color (str): The color of the plot.

    Returns:
        None
    """

    correlations = [df[var1].shift(lag).corr(df[var2]) for lag in lags]

    new_var1 = change_col_name(var1).capitalize()
    new_var2 = change_col_name(var2).capitalize()

    ax.plot(lags, correlations, marker='o', color=color)
    ax.set_title(f'"{new_var1}" and "{new_var2}"', fontsize=12)
    ax.set_xlabel('Lags (hours)', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--')


def lineplot_column(data: pd.DataFrame, 
                    x_col: str, 
                    y_col: str, 
                    title: str
                    ):
    """
    Plots a line plot for a specified column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column for the x-axis.
        y_col (str): The name of the column to plot on the y-axis.
        title (str): The title for the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=data, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('MW', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_daily_boxplots(data: pd.DataFrame, 
                        value_col: str
                        ):
    """
    Generates box plots for each day of the week to show the spread 
    and variability of the data. Includes variance annotations for each day.

    Args:
        data (pd.DataFrame): The input DataFrame with a datetime index and a value column.
        value_col (str): The name of the column containing the values to plot.

    Returns:
        None
    """

    data['day_of_week'] = data.index.day_name()

    # Calculate the variance for each day of the week
    variance_per_day = data.groupby('day_of_week')[value_col].var()

    # Sort the variance by the order of the week
    days_order = ['Monday', 
                  'Tuesday', 
                  'Wednesday', 
                  'Thursday', 
                  'Friday', 
                  'Saturday', 
                  'Sunday'
                  ]
    variance_per_day = variance_per_day.reindex(days_order)

    # Create the box plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='day_of_week', 
                     y=value_col, 
                     data=data, 
                     order=days_order,
                     )
    plt.title('Box Plots of Daily Distributions with Variance')
    plt.xlabel('Day of the Week')
    plt.ylabel(value_col)
    plt.grid(True)

    # Annotate the plot with variance values
    for i, day in enumerate(days_order):
        var_value = variance_per_day[day]
        if pd.notnull(var_value): 
            ax.text(i, data[value_col].max(), f'Var: {var_value:.2f}', 
                    ha='center', va='bottom', fontsize=10, color='red')
            
    data.drop(columns=['day_of_week'], inplace=True)

    plt.show()


def multiple_hist_plots(train_serie: pd.DataFrame, 
                        vali_serie: pd.DataFrame,
                        test_serie: pd.DataFrame,
                        col: str = None, 
                        num_cols: int=3, 
                        title: str=None
                        ):

    """
    Plots the frequency of values in time series.

    Args:
        train_serie (pandas.DataFrame): The time series to plot for the train set.
        vali_serie (pandas.DataFrame): The time series to plot for the validation set.
        test_serie (pandas.DataFrame): The time series to plot for the test set.
        col (str): The country name to plot (default: None).
        num_cols (int): Number of plots per country.
        title (str): The title of the plot (default: None).

    Returns:
        None
    """

    num_cols = num_cols
    num_rows = 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6.5 * num_rows))

    for i, ax in enumerate(axes.flat):
        if i < len(train_serie.columns):
            col = train_serie.columns[i]
            sns.histplot(train_serie[col], kde=True, ax=ax, color="blue", alpha=0.3, edgecolor=None)
            sns.histplot(vali_serie[col], kde=True, ax=ax, color="orange", alpha=0.3, edgecolor=None)
            sns.histplot(test_serie[col], kde=True, ax=ax, color="green",  alpha=0.3, edgecolor=None)
            ax.set_title(f"{col[:-7]}", fontsize=20)
            ax.legend(["Train", "Validation", "Test"], loc="upper right", fontsize=18)
        else:
            ax.axis('off')  
        ax.tick_params(axis='x', labelsize=18)  
        ax.tick_params(axis='y', labelsize=18) 
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_xlabel(None)

    if title is not None:
        fig.suptitle(title, fontsize=25)   
    plt.tight_layout()
    plt.show()
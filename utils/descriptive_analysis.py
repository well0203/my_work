import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def plot_missings(data, rotation=15):
    """
    Plots missing values.

    Args:
        data (pandas.DataFrame): The time series to plot.
        rotation (int): The rotation of the x-axis labels (default: 15).
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


def plot_seasonality(data, frequency='Month', 
                     hue='Year', num_cols=3, title=None):
    """
    Plots the seasonality of the time series.

    Args:
        data (pandas.DataFrame): The time series to plot.
        frequency (str): The frequency of the time series (default: 'Month').
        hue (str): Subsets of the data (default: 'Year').
        num_cols (int): Number of plots
        title (str): The title of the plot (default: None).
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


def hist_plots(data, col, num_cols=3, title=None):

    """
    Plots the frequency of values in time series.

    Args:
        data (pandas.DataFrame): The time series to plot.
        col (str): The country name to plot.
        num_cols (int): Number of plots per country.
        title (str): The title of the plot (default: None).
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


def corr_plot(data, annot=True, mask=True, title=None):
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


def count_outliers(col, extreme=False, verbose=False, if_return=False):
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
    

def periodograms(data, num_cols=3, max_ylim=4e8, title=None):
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
            
            ax.plot(f_per_year, np.abs(fft), drawstyle='steps-mid', color=sns.color_palette("muted")[i % len(sns.color_palette("muted"))], label="Amplitude")
            
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


def get_season(month
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


def stacked_bar_plot_per_season(data, 
                        season_order=['winter', 'spring', 'summer', 'autumn'], 
                        country=None):
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
    seasonal_avg = data.groupby('season').agg('mean').reset_index().sort_values(
        'season', key=lambda x: pd.Categorical(x, categories=season_order, ordered=True))

    melted_df = pd.melt(seasonal_avg, id_vars=['season'], value_vars=seasonal_avg.columns[1:])

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


def heatmap_plot(data, 
                 country, 
                 cmap=['Blues', 'Oranges', 'YlGnBu']):
    """
    Plots heatmaps for given time series data.

    Args:
        data (pandas.DataFrame): The DataFrame containing time series data.
        country (str): The name of the country to plot heatmaps for.
        cmap (list): A list of color maps for each heatmap (default: 
                    ['Blues', 'Oranges', 'YlGnBu']).

    Returns:
        None
    """
    # Clean and format column names
    """
    result_col_names = [' '.join(col.split('_')[2:-3]) if 'GB' in col and 'load' in col
                        else ' '.join(col.split('_')[1:-3]) if 'load' in col 
                        else ' '.join(col.split('_')[2:-1]) if 'GB' in col
                        else ' '.join(col.split('_')[1:-1]) for col in data.columns]
    """

    result_col_names = [change_col_name(col) for col in data.columns]

    # Extract hour and month for pivoting
    heatmap_data = data.copy()
    heatmap_data['hour'] = heatmap_data.index.hour
    heatmap_data['month'] = heatmap_data.index.month

    # Create pivot tables for heatmaps
    heatmaps = [heatmap_data.pivot_table(index='hour', columns='month', values=col, aggfunc='mean') for col in data.columns]
    
    # Create a 1x3 subplot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Iterate over heatmaps and axes to plot
    for i, (df, ax) in enumerate(zip(heatmaps, axs)):
        sns.heatmap(df, cmap=cmap[i], ax=ax, cbar_kws={'label': f'Avg {result_col_names[i]} (MW)'})
        ax.set_xlabel('Month', fontsize=12)
        if i == 0:
            ax.set_ylabel('Hour of Day', fontsize=12)
        else:
            ax.set_ylabel(None)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=30)
        ax.set_title(f'Avg "{result_col_names[i]}" by Hour and Month', fontsize=14)

    # Adjust layout to avoid overlap
    plt.suptitle(f'Heatmaps for {country}', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_correlations_between_2vars(df, var1, var2, ax, color, lags=range(1, 49)):
    """
    Plots correlation coefficients as a function of time lag.

    Args:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        var1 (str): The name of the first variable.
        var2 (str): The name of the second variable.
        lags (range): The range of lags tested (default: range(1, 49)).
        ax (matplotlib.axes.Axes): The axis object for the plot.
        color (str): The color of the plot.
    """
    correlations = [df[var1].shift(lag).corr(df[var2]) for lag in lags]

    new_var1 = change_col_name(var1).capitalize()
    new_var2 = change_col_name(var2).capitalize()

    ax.plot(lags, correlations, marker='o', color=color)
    ax.set_title(f'"{new_var1}" and "{new_var2}"', fontsize=12)
    ax.set_xlabel('Lags (hours)', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--')

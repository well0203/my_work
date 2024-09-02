import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
    """
    
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
        cbar_kws={"shrink": 0.75}  
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

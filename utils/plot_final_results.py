import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
from typing import List, Dict, Tuple


def get_x_positions_values(categories: List[str],
                           groups: List[str],
                           bar_positions: List[float]
                           ) -> Dict[Tuple[str, str], 
                                     Dict[str, float]]:
    """
    Get the x positions and values for the categories and groups.

    Args:
        categories (List[str]): The list of categories.
        groups (List[str]): The list of groups.
        bar_positions (List[float]): The positions of the bars.

    Returns:
        Dict[Tuple[str, str], Dict[str, float]]: A dictionary containing 
                                                 the x positions and values 
                                                 for each category and group.
    """
                    

    mapping = []
    num_groups = len(groups)

    for category_index in range(len(categories)):
        for group_index in range(num_groups):
            i = group_index * len(categories) + category_index  
            if i < len(bar_positions):  
                pos = bar_positions[i]
                category = categories[category_index]
                group = groups[group_index]
                mapping.append((pos, category, group))
            else:
                print(f"Index out of bounds for calculated index={i}")

    category_group_dict = {}
    for pos, category, group in mapping: 
        category_group_dict[(category, group)] = {'Position': pos} 

    return category_group_dict


def format_ticks(value: float,
                 tick_number: int,
                 decimal_places: int = 2
                 ) -> str:
    """Format tick labels to show two decimal points.

    Args:
        value (float): The value to format.
        tick_number (int): The index of the tick in the 
                           list of ticks being formatted.

    Returns:
        str: The formatted value.
    """

    return f"{value:.{decimal_places}f}"


def plot_results_comparison_models(data: pd.DataFrame, 
                                   ax: plt.Axes, 
                                   country: str = None, 
                                   eval_metric: str = 'RMSE'):
    """
    Plots lineplots for the results comparison between all models with 
    the specific evaluation metric for a specific country. 

    Args:
        data (pd.DataFrame): The dataframe to plot. 
        ax (matplotlib.axes.Axes): The axes to plot on.
        country (str): The country to plot (default: None).
        eval_metric (str): The evaluation metric to plot (default: 'RMSE').
    Returns:
        None
    """
    # Get models names
    models = data.columns.get_level_values('Model').unique().to_list()

    # Style markers and colors
    markers = ['o', 's', '*', '^', 'X', '<']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

    # Plot each model with its line and marker
    for idx, model in enumerate(models):
        
        # Subset data
        if country is None:
            subset = data.loc[:, (model, eval_metric)]
        else:
            # Subset data
            subset = data.loc[(country), (model, eval_metric)]

        # Plot
        ax.plot(subset.index, subset.values, 
                marker=markers[idx], color=colors[idx], label=model, 
                linestyle='-', linewidth=1.5, markersize=10)

    # Customize plot
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.set_ylabel(f'{eval_metric}', fontsize=16)
    ax.set_xticks([24, 96, 168])
    ax.set_xticklabels(['24h', '96h', '168h'])
    
    # Set font size for x and y ticks
    ax.tick_params(axis='x', labelsize=12)  
    ax.tick_params(axis='y', labelsize=12)

    # Set title
    if country is not None:
        ax.set_title(f'{country}', fontsize=16, fontweight='bold')
    else:
        ax.set_title(f'{eval_metric}', fontsize=16, fontweight='bold')


def plot_bar_comparison(data: pd.DataFrame, 
                        ax: plt.Axes, 
                        country: str = None, 
                        plot_x_axis: str = 'Pred_len',
                        eval_metric: str = 'RMSE',
                        plot_min_markers: bool = False,
                        plot_max_markers: bool = False,
                        palette: str = "CMRmap"):
    """
    Plots barplot for the results comparison between models with 
    the specific evaluation metric for a specific country. 

    Args:
        data (pandas.DataFrame): The dataframe to plot. 
        ax (matplotlib.axes.Axes): The axes to plot on.
        country (str): The country to plot (default: None).
        eval_metric (str): The evaluation metric to plot (default: 'RMSE').
        plot_x_axis (str): The x-axis to plot (default: 'Pred_len').
        plot_min_markers (bool): Whether to plot markers for the minimum values 
                                (default: False).
        plot_max_markers (bool): Whether to plot markers for the maximum values 
                                 (default: False).
        palette (str): The color palette to use (default: 'CMRmap').
    Returns:
        None
    """
    
    # Get models names
    models = data.columns.get_level_values('Model').unique().to_list()
    categories = data.index.get_level_values(plot_x_axis).unique().to_list()

    # Subset data
    if country is not None:
        subset = data.loc[(country), (slice(None), eval_metric)].reset_index()
    else:
        subset = data.loc[:, (slice(None), eval_metric)].reset_index()

    subset.columns = subset.columns.droplevel(1)
    
    subset = subset.melt(id_vars=plot_x_axis, 
                         value_vars=models, 
                         var_name='Model', 
                         value_name=eval_metric)

    # Plot
    bar_plot = sns.barplot(x=plot_x_axis, 
                y=eval_metric, 
                hue='Model', 
                data=subset, 
                palette=palette, 
                ax=ax) 
    
    # Plot markers of the max values
    if plot_min_markers or plot_max_markers:
    # Get the positions of each bar
    # It finds the center of each bar
        bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in bar_plot.patches]
        xs_dict = get_x_positions_values(categories, models, bar_positions)

    if plot_max_markers:
        max_vals = subset.loc[subset.groupby(plot_x_axis)[eval_metric].idxmax()]
        for c, m, max_,  in zip(max_vals[plot_x_axis], max_vals['Model'], max_vals[eval_metric]):
            x = xs_dict[(c, m)]['Position']
            ax.plot(x, max_, marker='v', color='red')

    if plot_min_markers:
        min_vals = subset.loc[subset.groupby(plot_x_axis)[eval_metric].idxmin()]
        for c, m, min_, in zip(min_vals[plot_x_axis], min_vals['Model'], min_vals[eval_metric]):
            x = xs_dict[(c, m)]['Position']
            ax.plot(x, min_, marker='o', color='black')


    # Style
    if country is not None:
        ax.set_title(f"{country}", fontsize=16, fontweight='bold')
        y_label = eval_metric
    else:
        ax.set_title(f"{eval_metric}", fontsize=16, fontweight='bold')
        y_label = ''

    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xlabel('')
    ax.legend_.remove()
    
    # Set custom x-tick labels
    if plot_x_axis == 'Pred_len':
        x_ticks = ax.get_xticks()  
        tick_labels = ['24h', '96h', '168h']  
        ax.set_xticks(x_ticks)  
        ax.set_xticklabels(tick_labels) 
        rotation = 0
    else:
        rotation = 15
    # Set font size for x and y ticks
    ax.tick_params(axis='x', labelsize=12, rotation=rotation)  
    ax.tick_params(axis='y', labelsize=12) 
    

def plot_results_models_multiple_countries(data: pd.DataFrame, 
                                           function: str = "lines", 
                                           eval_metric: str = 'RMSE', 
                                           palette: str = "CMRmap",
                                           plot_min_markers: bool = False,
                                           plot_max_markers: bool = False,
                                           decimal_places: int = 2):
    """
    Creates a grid of subplots for each specified country over pred_lens and models 
    for a specific evaluation metric.

    Args:
        data (pandas.DataFrame): The dataframe to plot. It has to be in multiindex 
                                 and multicolumn format (two levels index and columns).
        function (str): The function to plot. Options: 'lines', 'bars'
                        (default: 'lines').
        eval_metric (str): The evaluation metric to plot (default: 'RMSE').
        palette (str): The color palette to use (default: 'CMRmap').
        plot_min_markers (bool): Whether to plot markers for the minimum values 
                                 (default: False).
        plot_max_markers (bool): Whether to plot markers for the maximum values 
                                 (default: False).
        decimal_places (int): The number of decimal places to round to (default: 2).
    Returns:
        None
    """

    # 0 = Country
    countries = data.index.get_level_values(0).unique().to_list()


    num_countries = len(countries)

    fig = plt.figure(figsize=(18, 8)) #, tight_layout=True)
    spec = gridspec.GridSpec(ncols=8, 
                             nrows=2, 
                             figure=fig, 
                             wspace=0.2, 
                             hspace=0.4)

    ax1 = fig.add_subplot(spec[0,1:3])
    ax2 = fig.add_subplot(spec[0,3:5], sharey=ax1)
    ax3 = fig.add_subplot(spec[0,5:7], sharey=ax1)
    ax4 = fig.add_subplot(spec[1,2:4], sharey=ax1)
    ax5 = fig.add_subplot(spec[1,4:6], sharey=ax1)

    axes = [ax1, ax2, ax3, ax4, ax5]

    for idx, ax in enumerate(axes):
        # Only plot for available countries
        if idx < num_countries:  
            country = countries[idx]
            if function == "lines":
                plot_results_comparison_models(data, 
                                               ax=ax, 
                                               country=country, 
                                               eval_metric=eval_metric)
            elif function == "bars":
                plot_bar_comparison(data, 
                                    ax=ax, 
                                    country=country, 
                                    eval_metric=eval_metric,
                                    plot_min_markers=plot_min_markers,
                                    plot_max_markers=plot_max_markers, 
                                    palette=palette)

            # Format y-ticks labels
            y_min, y_max = ax.get_ylim() 
            ax.set_yticks(np.linspace(y_min, y_max, 4)) 
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, 
                              tick_number: 
                              format_ticks(value, 
                                           tick_number, 
                                           decimal_places=decimal_places)))


            # Remove y-label except for 0 and 3rd plot
            if idx != 0 and idx != 3:
                ax.set_ylabel('')
                # Hide y-tick labels 
                ax.tick_params(labelleft=False)

    # Position legend below the entire row of plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, 
               labels, 
               loc='lower center', 
               ncol=4, 
               fontsize=16, 
               bbox_to_anchor=(0.5, -0.05))

    plt.show()


def plot_comparison_for_metrics(data: pd.DataFrame, 
                                plot_x_axis: str = 'Pred_len',
                                plot_type: str = "bars",
                                palette: str = "CMRmap",
                                plot_min_markers: bool = False,
                                plot_max_markers: bool = False,
                                decimal_places: int = 2,
                                percentage: bool = False):
    """
    Calls the plot_bar_comparison function to plot two metrics side by side.

    Args:
        data (pandas.DataFrame): The dataframe to plot. 
        plot_x_axis (str): The x-axis to plot (default: 'Pred_len').
        plot_type (str): The type of plot to use (default: 'bars').
                         Possible values: 'bars', 'lines'.
        palette (str): The color palette to use (default: 'CMRmap').
        plot_min_markers (bool): Whether to plot markers for the minimum values 
                                (default: False).
        plot_max_markers (bool): Whether to plot markers for the maximum values 
                                 (default: False).
        decimal_places (int): The number of decimal places to round to 
                       (default: 2).
    Returns:
        None
    """
    metrics = data.columns.get_level_values('Metrics').unique().to_list()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    if plot_type == 'bars':
        # Plot both evaluation metrics
        for idx, metric in enumerate(metrics):
            plot_bar_comparison(data, 
                                axs[idx], 
                                country=None, 
                                plot_x_axis=plot_x_axis, 
                                eval_metric=metric,
                                plot_min_markers=plot_min_markers, 
                                plot_max_markers=plot_max_markers, 
                                palette=palette)
    elif plot_type == 'lines':
        # Plot both evaluation metrics
        for idx, metric in enumerate(metrics):
            plot_results_comparison_models(data, 
                                           axs[idx], 
                                           country=None, 
                                           eval_metric=metric)
    
    # Format y-ticks labels
    y_min, y_max = axs[0].get_ylim() 
    axs[0].set_yticks(np.linspace(y_min, y_max, 4)) 
    axs[0].yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, 
            tick_number: format_ticks(
                value, 
                tick_number, 
                decimal_places=decimal_places)))

    # Position legend below the entire row of plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, 
               labels, 
               loc='lower center', 
               ncol=4, 
               fontsize=16, 
               bbox_to_anchor=(0.5, -0.15))

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
def percent_formatter(x, pos):

    return f"{x:.0f}%"


def plot_barplots(data: pd.DataFrame, 
                  x_col: str, 
                  col_name: str = 'Metrics', 
                  palette: str = 'CMRmap',
                  decimal_places: int = 2,
                  percentage: bool = False):
    """
    Plots two bar plots side-by-side using Seaborn.
    
    Args:
    - data (pd.DataFrame): DataFrame to plot.
    - x_col (str): The column name for the x-axis.
    - col_name (str): The column name for the y-axis.
    - palette (str): The color palette to use.
    - percentage (bool): Whether to plot y-axis as percentage 
                        (default: False).

    
    Returns:
    - None
    """

    y_cols = list(data[col_name].unique())
    fig, axes = plt.subplots(1, len(y_cols), figsize=(8, 4), sharey=True)
    
    for ax, y_col in zip(axes, y_cols):
        subset = data[data[col_name] == y_col]
        sns.barplot(hue=x_col, y='Value', data=subset, ax=ax, palette=palette)#, legend=False)
        ax.set_title(f'{y_col}', fontsize=16, fontweight='bold')
        """
        # Annotate each bar with its value
        for bar in ax.patches:
            height = bar.get_height()
            
            if not np.isnan(height):  
                ax.annotate(f'{height}',#:.{decimal_places}f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
        """
        y_min, y_max = ax.get_ylim() 
        ax.set_yticks(np.linspace(y_min, y_max, 4)) 
        ax.tick_params(axis='y', labelsize=12)

        ax.set_ylabel('')
        ax.legend_.remove()
        
        ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, 
                              tick_number: 
                              format_ticks(value, 
                                           tick_number, 
                                           decimal_places=decimal_places)))
        
        if percentage:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(percent_formatter)) 
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, 
               labels, 
               loc='lower center', 
               ncol=5, 
               fontsize=16, 
               bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    plt.show()
    
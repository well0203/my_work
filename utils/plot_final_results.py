import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def format_ticks(value,
                 tick_number,
                 decimal_places=2
                 ) -> str:
    """Format tick labels to show two decimal points.

    Args:
        value (float): The value to format.
        tick_number (int): The index of the tick in the list of ticks being formatted

    Returns:
        str: The formatted value.
    """

    return f"{value:.{decimal_places}f}"


def plot_results_comparison_models(data, 
                                   ax, 
                                   country='DE', 
                                   eval_metric='RMSE'):
    """
    Plots lineplots for the results comparison between all models with 
    the specific evaluation metric for a specific country. 

    Args:
        data (pandas.DataFrame): The dataframe to plot. 
        ax (matplotlib.axes.Axes): The axes to plot on.
        country (str): The country to plot (default: 'DE').
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
    ax.set_title(f'{country}', fontsize=16, fontweight='bold')


def plot_bar_comparison(data, 
                        ax, 
                        country=None, 
                        plot_x_axis='Pred_len',
                        eval_metric='RMSE',
                        palette="CMRmap"):
    """
    Plots barplot for the results comparison between models with 
    the specific evaluation metric for a specific country. 

    Args:
        data (pandas.DataFrame): The dataframe to plot. 
        ax (matplotlib.axes.Axes): The axes to plot on.
        country (str): The country to plot (default: 'DE').
        eval_metric (str): The evaluation metric to plot (default: 'RMSE').
        palette (str): The color palette to use (default: 'CMRmap').
    Returns:
        None
    """

    # Get models names
    models = data.columns.get_level_values('Model').unique().to_list()

    # Subset data
    if country is not None:
        subset = data.loc[(country), (slice(None), eval_metric)].reset_index()
    else:
        subset = data.loc[:, (slice(None), eval_metric)].reset_index()

    subset.columns = subset.columns.droplevel(1)
    
    subset = subset.melt(id_vars=plot_x_axis, value_vars=models, var_name='Model', value_name=eval_metric)

    # Plot
    sns.barplot(x=plot_x_axis, y=eval_metric, hue='Model', data=subset, palette=palette, ax=ax) 

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
    

def plot_results_models_multiple_countries(data, 
                                           function="lines", 
                                           eval_metric='RMSE', 
                                           palette="CMRmap",
                                           decimal_places=2):
    """
    Creates a grid of subplots for each specified country over pred_lens and models for a specific evaluation metric.

    Args:
        data (pandas.DataFrame): The dataframe to plot. It has to be in multiindex format.
        function (str): The function to plot. Options: 'lines', 'bars'
                        (default: 'lines').
        eval_metric (str): The evaluation metric to plot (default: 'RMSE').
        palette (str): The color palette to use (default: 'CMRmap').
        decimal_places (int): The number of decimal places to round to (default: 2).
    Returns:
        None
    """

    # 0 = Country
    countries = data.index.get_level_values(0).unique().to_list()


    num_countries = len(countries)

    fig = plt.figure(figsize=(18, 8)) #, tight_layout=True)
    spec = gridspec.GridSpec(ncols=8, nrows=2, figure=fig, wspace=0.2, hspace=0.4)

    ax1 = fig.add_subplot(spec[0,1:3])
    ax2 = fig.add_subplot(spec[0,3:5], sharey=ax1)
    ax3 = fig.add_subplot(spec[0,5:7], sharey=ax1)
    ax4 = fig.add_subplot(spec[1,2:4], sharey=ax1)
    ax5 = fig.add_subplot(spec[1,4:6], sharey=ax1)

    axes = [ax1, ax2, ax3, ax4, ax5]

    for idx, ax in enumerate(axes):
        if idx < num_countries:  # Only plot for available countries
            country = countries[idx]
            if function == "lines":
                plot_results_comparison_models(data, ax=ax, country=country, eval_metric=eval_metric)
            elif function == "bars":
                plot_bar_comparison(data, ax=ax, country=country, eval_metric=eval_metric, palette=palette)

            # Format y-ticks labels
            y_min, y_max = ax.get_ylim() 
            ax.set_yticks(np.linspace(y_min, y_max, 4)) 
            #ax.yaxis.set_major_formatter(FuncFormatter(format_ticks)) 
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: format_ticks(value, tick_number, decimal_places=decimal_places)))


            # Remove y-label except for 0 and 3rd plot
            if idx != 0 and idx != 3:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)  # Hide y-tick labels without clearing them

    # Position legend below the entire row of plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.05))

    plt.show()


def plot_comparison_for_metrics(data, 
                                plot_x_axis='Pred_len',
                                palette="CMRmap",
                                decimal_places=2):
    """
    Calls the plot_bar_comparison function to plot two metrics side by side.

    Args:
        data (pandas.DataFrame): The dataframe to plot. 
        palette (str): The color palette to use (default: 'CMRmap').
    Returns:
        None
    """
    metrics = data.columns.get_level_values('Metrics').unique().to_list()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Plot both evaluation metrics
    for idx, metric in enumerate(metrics):
        plot_bar_comparison(data, 
                            axs[idx], 
                            country=None, 
                            plot_x_axis=plot_x_axis, 
                            eval_metric=metric, 
                            palette=palette)
    
    # Format y-ticks labels
    y_min, y_max = axs[0].get_ylim() 
    axs[0].set_yticks(np.linspace(y_min, y_max, 4)) 
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda value, tick_number: format_ticks(value, tick_number, decimal_places=decimal_places)))

    # Position legend below the entire row of plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, -0.15))

    # Adjust layout
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns


def plot_results_comparison_models(data, ax, country='DE', eval_metric='RMSE'):
    # (Your existing plotting code goes here)
    models = data.columns.get_level_values('Model').unique().to_list()
    markers = ['o', 's', '*', '^', 'X', '<']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

    for idx, model in enumerate(models):
        subset = data.loc[(country), (model, eval_metric)]
        ax.plot(subset.index, subset.values, 
                marker=markers[idx], color=colors[idx], label=model, 
                linestyle='-', linewidth=1.5, markersize=10)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.set_xlabel('Prediction Length', fontsize=10)
    ax.set_ylabel(f'{eval_metric}', fontsize=10)
    ax.set_xticks([24, 96, 168])
    ax.set_xticklabels(['24h', '96h', '168h'])
    ax.set_title(f'{country}', fontsize=12, fontweight='bold')


def plot_bar_comparison(data, ax, country='DE', eval_metric='RMSE', palette="CMRmap"):
    # (Your existing bar plotting code goes here)
    models = data.columns.get_level_values('Model').unique().to_list()
    subset = data.loc[(country), (slice(None), eval_metric)].reset_index()
    subset.columns = subset.columns.droplevel(1)
    subset = subset.melt(id_vars=['Pred_len'], value_vars=models, var_name='Model', value_name=eval_metric)

    # Plot
    sns.barplot(x='Pred_len', y=eval_metric, hue='Model', data=subset, palette=palette, ax=ax) 

    # Style
    ax.set_title(f"{country}", fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction Length', fontsize=10)
    ax.set_ylabel(eval_metric, fontsize=10)
    ax.legend_.remove()

    # Set custom x-tick labels
    x_ticks = ax.get_xticks()  
    tick_labels = ['24h', '96h', '168h']  
    ax.set_xticks(x_ticks)  
    ax.set_xticklabels(tick_labels)     


def plot_results_models_multiple_countries(data, function="lines", countries=['DE', 'ES', 'FR', 'GB', 'IT'], eval_metric='RMSE', palette="CMRmap"):
    """
    Creates a grid of subplots for each specified country over pred_lens and models for a specific evaluation metric.

    Args:
        data (pandas.DataFrame): The dataframe to plot. 
        function (str): The function to plot (default: 'lines').
        countries (list of str): List of countries to plot.
        eval_metric (str): The evaluation metric to plot (default: 'RMSE').
    Returns:
        None
    """
    num_countries = len(countries)

    # Create a figure with gridspec layout
    fig = plt.figure(figsize=(18, 8), tight_layout=True)
    spec = gridspec.GridSpec(ncols=8, nrows=2, figure=fig)

    ax1 = fig.add_subplot(spec[0,1:3])
    ax2 = fig.add_subplot(spec[0,3:5], sharey=ax1)
    ax3 = fig.add_subplot(spec[0,5:7], sharey=ax1)
    ax4 = fig.add_subplot(spec[1,2:4], sharey=ax1)
    ax5 = fig.add_subplot(spec[1,4:6], sharey=ax1)

    axes = [ax1, ax2, ax3, ax4, ax5]

    plt.suptitle(f'Comparison of Models with {eval_metric}', fontsize=16, y=1.01) #, fontweight='bold', y=1.05)

    for idx, ax in enumerate(axes):
        if idx < num_countries:  # Only plot for available countries
            country = countries[idx]
            if function == "lines":
                plot_results_comparison_models(data, ax=ax, country=country, eval_metric=eval_metric)
            elif function == "bars":
                plot_bar_comparison(data, ax=ax, country=country, eval_metric=eval_metric, palette=palette)

            # Remove y-label except for 0 and 3rd plot
            if idx != 0 and idx != 3:
                ax.set_ylabel('')
                ax.set_yticklabels([])  # Remove y-tick labels

    # Position legend below the entire row of plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Models", loc='lower center', ncol=4, fontsize=10, title_fontsize='11', bbox_to_anchor=(0.5, -0.1))

    plt.show()


# 1 line of plots
"""
def plot_results_models_multiple_countries(data, function="lines", countries=['DE', 'ES', 'FR', 'GB', 'IT'], eval_metric='RMSE'):

    Creates a row of subplots for each specified country over pred_lens and models for specific evaluation metric.

    Args:
        data (pandas.DataFrame): The dataframe to plot. 
        function=(str): The function to plot (default: 'lines').
        countries (list of str): List of countries to plot.
        eval_metric (str): The evaluation metric to plot (default: 'RMSE').
    Returns:
        None

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    plt.suptitle(f'Comparison of Models with {eval_metric}', fontsize=16, fontweight='bold', y=1.05)

    for idx, (ax, country) in enumerate(zip(axes, countries)):
        if function == "lines":
            plot_results_comparison_models(data, ax=ax, country=country, eval_metric=eval_metric)
        elif function == "bars":
            plot_bar_comparison(data, ax=ax, country=country, eval_metric=eval_metric)

        # Remove y-label for all subplots except the first
        if idx > 0:
            ax.set_ylabel('')  # Clear y-label for non-first subplots


    # Position legend below the entire row of plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Models", loc='lower center', ncol=4, fontsize=10, title_fontsize='11', bbox_to_anchor=(0.5, -0.1))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title and bottom legend
    plt.show()
"""

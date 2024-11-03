def plot_bar_comparison(data, 
                        ax, 
                        country='DE', 
                        eval_metric='RMSE', 
                        palette="CMRmap"):
    """
    Plots barplots for the results comparison between specified models with 
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
    subset = data.loc[(country), (slice(None), eval_metric)].reset_index()
    subset.columns = subset.columns.droplevel(1)
    subset = subset.melt(id_vars=['Pred_len'], value_vars=models, var_name='Model', value_name=eval_metric)

    # Plot
    sns.barplot(x='Pred_len', y=eval_metric, hue='Model', data=subset, palette=palette, ax=ax) 

    # Style

    ax.set_title(f"{country}", fontsize=16, fontweight='bold')
    ax.set_ylabel(eval_metric, fontsize=16)
    ax.set_xlabel('')
    ax.legend_.remove()
    
    # Set custom x-tick labels
    x_ticks = ax.get_xticks()  
    tick_labels = ['24h', '96h', '168h']  
    ax.set_xticks(x_ticks)  
    ax.set_xticklabels(tick_labels) 

    # Set font size for x and y ticks
    ax.tick_params(axis='x', labelsize=12)  
    ax.tick_params(axis='y', labelsize=12)    
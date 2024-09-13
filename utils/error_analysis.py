import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_average_results(paths):
    """
    Average predictions and true values across multiple experiments.

    Args:
        paths (list): List of paths to load predictions and true values from.

    Returns:
        avg_pred (np.ndarray): Averaged predictions.
        avg_true (np.ndarray): Averaged true values.
    """
    preds_list = []
    trues_list = []
    
    for path in paths:
        pred = np.load(os.path.join(path, "pred.npy"))  
        true = np.load(os.path.join(path, "true.npy"))  
        
        preds_list.append(pred)
        trues_list.append(true)

    # Average
    avg_pred = np.mean(np.stack(preds_list), axis=0)  # Shape (N, L, C)
    avg_true = np.mean(np.stack(trues_list), axis=0)  # Shape (N, L, C)

    return avg_pred, avg_true


def plot_results(avg_pred, avg_true, columns, loss_type, pred_len, color):
    """
    Plots predictions and true values for each column in a hexbin plot.

    Args:
        avg_pred (np.ndarray): Averaged predictions.
        avg_true (np.ndarray): Averaged true values.
        columns (list): List of column names.
        loss_type (str): Type of loss.
        pred_len (int): Length of the prediction.
        color (str): Color of the plot.

    Returns:
        None
    """

    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(20, 3))

    for i, col in enumerate(columns):
        pred_flat = avg_pred[:, :, i].flatten()
        true_flat = avg_true[:, :, i].flatten()

        axes[i].set_xlabel(f"Predicted '{col}'", fontsize=10)
        cax = axes[i].hexbin(pred_flat, true_flat, gridsize=50, cmap=color)
        fig.colorbar(cax, ax=axes[i], location="top")

        # Add 45° line
        min_val = min(np.min(pred_flat), np.min(true_flat))
        max_val = max(np.max(pred_flat), np.max(true_flat))
        axes[i].set_xlim([min_val, max_val])
        axes[i].set_ylim([min_val, max_val])

        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        l_join = col.split('_')[1:3]
        print(f"{(' ').join(l_join):<20} min value: {min_val:<10.2f} max value: {max_val:<10.2f}")

    
    axes[0].set_ylabel("True", fontsize=10)
    plt.suptitle(f"Loss: {loss_type}, Pred_len: {pred_len}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_error_results(model_name, loss_type, seq_len, pred_len, itr, columns, color):

    """
    Plots predictions and true values for a given model and loss type.

    Args:
        model_name (str): Name of the model.
        loss_type (str): Type of loss.
        seq_len (int): Length of the sequence.
        pred_len (int): Length of the prediction.
        ite (int): Number of experiment iterations.
        columns (list): List of column names.

    Returns:
        None
    """

    base_path = "./results_loss_unscaled"
    
    paths = [
        os.path.join(
            base_path, 
            f"DE_{seq_len}_{pred_len}_loss_choice_for_DE_{model_name}_custom_ftM_sl{seq_len}_ll5_pl{pred_len}_dm512_nh8_el2_dl1_df2048_fc5_ebtimeF_dtTrue_loss{loss_type}_Exp_{i}"
        ) for i in range(itr)
    ]
    
    avg_pred, avg_true = load_and_average_results(paths)
    
    plot_results(avg_pred, avg_true, columns, loss_type, pred_len, color)

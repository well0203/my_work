import subprocess
import optuna
import os

# Function to run the experiment and capture the output accuracy or relevant metric
def run_experiment(batch_size, lr, n_heads, pred_len=10, seq_len=20, model_id_name="test", model="Informer"):
    # Paths to files and data
    data_path = os.getcwd() + "/datasets/"
    script_path = "./PatchTST-main/PatchTST_supervised/run_longExp.py"
    log_file_path = f"logs/LongForecasting/{model}_{model_id_name}_{seq_len}_{pred_len}.log"

    # Build the command with the current hyperparameters
    command = f"""
    python {script_path} \
      --random_seed 2021 \
      --is_training 1 \
      --root_path "{data_path}" \
      --data_path "GB_data_small.csv" \
      --model_id {model_id_name}_{seq_len}_{pred_len} \
      --model "{model}" \
      --data "custom" \
      --features M \
      --seq_len {seq_len} \
      --label_len 5 \
      --pred_len {pred_len} \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 5 \
      --dec_in 5 \
      --c_out 5 \
      --des 'Exp' \
      --train_epochs 2 \
      --patience 1 \
      --patch_len 3 \
      --stride 2 \
      --n_heads {n_heads} \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --inverse \
      --overlapping_windows \
      --itr 1 --batch_size {batch_size} --learning_rate {lr}
    """

    # Run the command and capture output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output = []
    for line in process.stdout:
        output.append(line)
        print(line, end="")  # Output in real-time

    process.wait()

    # Parse the output for any relevant metric (assuming accuracy is printed like "accuracy: 0.9")
    accuracy = None
    for line in output:
        if "accuracy" in line:  # Modify this depending on your output
            accuracy = float(line.split(":")[1].strip())
            break

    # Return the metric to Optuna (higher is better)
    return accuracy if accuracy else 0.0

# Objective function for Optuna tuning
def objective(trial):
    # Define the hyperparameter search space
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    n_heads = trial.suggest_int("n_heads", 8, 16)

    # Run the experiment with the given hyperparameters
    accuracy = run_experiment(batch_size=batch_size, lr=lr, n_heads=n_heads)

    # Optuna will try to maximize this metric
    return accuracy

# Main function to run the optimization
if __name__ == "__main__":
    # Create an Optuna study to maximize accuracy
    study = optuna.create_study(direction="minimize")

    # Run the optimization (you can specify n_trials or let it run indefinitely)
    study.optimize(objective, n_trials=20)

    # Print the best found hyperparameters
    print("Best hyperparameters found:")
    print(study.best_trial.params)

    # Optionally, save the study results to a file
    study.trials_dataframe().to_csv("optuna_tuning_results.csv")

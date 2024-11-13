import numpy as np
import os
from utils.metrics import metric
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

plt.switch_backend('agg')

def running_time(start_time: float, end_time: float):
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

def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=True, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def vali(args, accelerator, model, vali_data, vali_loader, criterion):
    file_path = "results.txt"

    with open(file_path, "w") as file:
    # Convert the Namespace object to a dictionary
        args_dict = vars(args)

        # Write each attribute to the file
        for key, value in args_dict.items():
            file.write(f"{key} = {value}\n")

    total_loss = []
    model.eval()
    with torch.no_grad():
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            total_loss.append(loss.item())

    total_loss = np.average(total_loss)

    model.train()
    return total_loss


def test(args, accelerator, model, test_data, test_loader, criterion, setting):

    checkpoint_path = os.path.join('./checkpoints/' + setting + '-' + args.model_comment, 'checkpoint.pth')
    
    if accelerator is not None:
        accelerator.print('loading model...')

        # 1. Unwrap the model if using Accelerate
        # We do not need distributed training functionality for testing

        model = accelerator.unwrap_model(model)
        # 2. Load the model weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=accelerator.device))
    else:
        print('loading model...')
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    folder_path = f'./test_results/{args.model}/' + setting + '/'
    os.makedirs(folder_path, exist_ok=True)

    preds = []
    trues = []

    total_loss = []
    model.eval()
    with torch.no_grad():
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # Tensors without gradient computation to reduce memory consumption
            pred = outputs.detach()
            true = batch_y.detach()

            # Needs GPU
            loss = criterion(pred, true)

            total_loss.append(loss.item())

            # We do not need GPU anymore
            pred = pred.cpu().numpy()
            true = true.cpu().numpy()

            preds.append(pred)
            trues.append(true)

            # First observation from a batch every 100 test batches
            if i % 100 == 0:
                input = batch_x.detach().cpu().numpy()

                # (24, 20, 1)
                # Print first observation from the batch and flatten third dim
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    preds = np.array(preds)
    trues = np.array(trues)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # print("pred.shape after reshape", pred.shape)

    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    p = "./results_transformers"
    if not os.path.exists(p):
        os.makedirs(p)

    #folder_path = f'./results/{self.args.model}/' + '/'
    folder_path_1 = p + '/' + f'{args.model}/' + setting + '/' #f'{args.data}/'
    if not os.path.exists(folder_path_1):
        os.makedirs(folder_path_1, exist_ok = True)

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print('Scaled mse:{}, rmse:{}, mae:{}, rse:{}'.format(mse, rmse, mae, rse))

    f = open("result_long_term_forecast.txt", 'a')
    f.write(setting + "  \n")
    f.write('Scaled mse:{}, rmse:{}, mae:{}, rse:{}'.format(mse, rmse, mae, rse))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path_1 + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path_1 + 'pred.npy', preds)
    np.save(folder_path_1 + 'true.npy', trues)

    total_loss = np.average(total_loss)

    model.train()
    return total_loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    # with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
    with open('./Time-LLM/dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:

        content = f.read()
    return content

"""
def plot_train_val_loss(args, train_loss, val_loss, lr=None):
    
    Plots training and validation loss with optional learning rate values.
    
    Args:
    train_loss (list): List containing training loss values for each epoch.
    val_loss (list): List containing validation loss values for each epoch.
    lr_values (list, optional): List containing learning rate values for each epoch.
    
    Returns:
    legend_labels (list): List containing legend labels for the plot.
    
    epochs = range(1, len(train_loss) + 1)
    legend_labels = []

    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    legend_labels.append('Training Loss')

    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    legend_labels.append('Validation Loss')

    # plt.title(f'Training and Validation Loss for Learning rate {lr}')
    plt.title(f'{args.data} dataset with pred_len {args.pred_len} and {args.percent} % of data. {args.llm_model} with {args.llama_layers} layers. LR {args.learning_rate}, batch size {args.batch_size}, d_model {args.d_model}, d_ff {args.d_ff} ')
    file_name = f'{args.data}_with_pred_len_{args.pred_len}_{args.percent}_data._{args.llm_model}_with_{args.llama_layers}_layers._LR_{args.learning_rate}_batch_size_{args.batch_size}_d_model_{args.d_model}_d_ff_{args.d_ff}'

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.join(file_name)
    plt.show()

    #return legend_labels
"""


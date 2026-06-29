import numpy as np
import torch
import json
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from cross_models.moe_models.primary_shared_expert import Expert

def is_expert(module):
    return isinstance(module, Expert)

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 
                     4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 
                     8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 
                     10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 
                     20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def a_law_compress(x, A=0.25):
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    compressed = np.where(abs_x < 1 / A, 
                          A * abs_x / (np.e + np.log(A)),
                          (1 + np.log(A * abs_x)) / (np.e + np.log(A)))
    
    return sign_x * compressed

def save_plots(batch_x, mask_info, pred, step, save_path):
    batch_size, dim_size, len_size = batch_x.shape
    _, _, sub_len_size = pred.shape

    folder_path = os.path.join(save_path, f'{step}')
    os.makedirs(folder_path, exist_ok=True)

    np.save(f'{folder_path}/batchx.npy', batch_x)
    np.save(f'{folder_path}/maskinfo.npy', mask_info)
    np.save(f'{folder_path}/pred.npy', pred)
    
    color_bk = '#649BC9'
    color_gt = '#FFAC52' #6BC6B9
    color_pred = '#F95C49'
    
    for batch in range(batch_size):
        for dim in range(dim_size):
            plt.figure(figsize=(12, 4))
            plt.plot(batch_x[batch, dim, :], label='ground truth', color=color_bk)
            
            mask_positions = np.where(mask_info[batch, dim, :])[0]
            mask_gt = batch_x[batch, dim, mask_positions]
            
            plt.scatter(mask_positions, batch_x[batch, dim, mask_positions], color=color_gt, label='masked', s=2)
            if len(mask_positions) > 1:
                start_idx = 0
                for j in range(1, len(mask_positions)):
                    if mask_positions[j] != mask_positions[j - 1] + 1:
                        plt.plot(mask_positions[start_idx:j], mask_gt[start_idx:j], color=color_gt, linewidth=2)
                        start_idx = j

            plt.scatter(mask_positions, pred[batch, dim, :], color=color_pred, label='predict', s=10)
            if len(mask_positions) > 1:
                start_idx = 0
                for j in range(1, len(mask_positions)):
                    if mask_positions[j] != mask_positions[j - 1] + 1:
                        plt.plot(mask_positions[start_idx:j], pred[batch, dim, start_idx:j], color=color_pred, linewidth=1)
                        start_idx = j

                # plt.plot(mask_positions[start_idx:], pred[batch, dim, start_idx:], color=color_pred, linewidth=1)

            plt.legend()
            plt.xlabel('time')
            plt.ylabel('uV')

            file_path = os.path.join(folder_path, f'{batch}_{dim}.png')
            plt.savefig(file_path)
            plt.close()


def save_confusion_matrix(confusion_matrix: torch.Tensor, folder_path: str, batch, dataset, part):
    confusion_matrix_np = confusion_matrix.numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_np, annot=True, fmt=".2f", cmap="Blues")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()

    file_path = os.path.join(folder_path, f'cm_{part}_{batch}_{dataset}.png')
    plt.savefig(file_path)
    plt.close()


def parse_bool_type(bool_type_str):
    if isinstance(bool_type_str, str):
        if bool_type_str.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif bool_type_str.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise ValueError('Boolean value expected.')


def parse_amp_type(amp_type_str):
    if amp_type_str == 'fp16':
        return torch.float16
    elif amp_type_str == 'bf16':
        return torch.bfloat16
    else:
        raise ValueError("Invalid amp_type. Choose either 'fp16' or 'bf16'.")


def default_serializer(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)  # Convert torch.dtype to string
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def get_3d_pos(max_range, args):
    if args.data in ["Stage1", "TUEV", "TUAB", "BCICIV-2B", "BCICIV-2B-SC", "BCICIV-2B-SC-TTS", "BCIC2020-3", 
                     "Mumtaz", "MentalArithmetic", "PhysioP300", "PhysioP300-TTS", "SEED-V", "SEED-V-bipolar", "FACED"]:
        montage = mne.channels.make_standard_montage('standard_1020')
        positions = montage.get_positions()['ch_pos']
        
        if args.cls_only and args.data in ["TUEV", "TUAB"]:
            double_banana_pairs = [
                ('Fp1', 'F7'), 
                ('F7', 'T3'),
                ('T3', 'T5'), 
                ('T5', 'O1'),
                ('Fp2', 'F8'), 
                ('F8', 'T4'),
                ('T4', 'T6'), 
                ('T6', 'O2'),
                # ('A1', 'T3'),
                ('T3', 'C3'),
                ('C3', 'Cz'), 
                ('Cz', 'C4'),
                ('C4', 'T4'),
                # ('T4', 'A2'),
                ('Fp2', 'F3'), 
                ('F3', 'C3'),
                ('C3', 'P3'), 
                ('P3', 'O1'),
                ('Fp2', 'F4'), 
                ('F4', 'C4'),
                ('C4', 'P4'), 
                ('P4', 'O2'),
            ]
        elif args.data in ["BCICIV-2B"]:
            double_banana_pairs = [
                ('C3', 'Cz'), 
                ('Cz', 'C4'),
            ]
        elif args.data in ["BCICIV-2B-SC", "BCICIV-2B-SC-TTS"]:
            double_banana_pairs = [
                "C3", 
                "Cz", 
                "C4",
            ]
        elif args.data in ["SEED-V"]:
            double_banana_pairs = [
                "Fp1", 
                "Fp2",
                "F7",
                "F3",
                "F4",
                "F8",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "P7",
                "P3",
                "P4",
                "P8",
                "O1",
                "O2",
            ]
        else:
            double_banana_pairs = [
                ('Fp1', 'F7'), 
                ('F7', 'T3'),
                ('T3', 'T5'), 
                ('T5', 'O1'),
                ('Fp2', 'F8'), 
                ('F8', 'T4'),
                ('T4', 'T6'), 
                ('T6', 'O2'),
                # ('A1', 'T3'),
                ('T3', 'C3'),
                ('C3', 'Cz'), 
                ('Cz', 'C4'),
                ('C4', 'T4'),
                # ('T4', 'A2'),
                ('Fp2', 'F3'), 
                ('F3', 'C3'),
                ('C3', 'P3'), 
                ('P3', 'O1'),
                ('Fp2', 'F4'), 
                ('F4', 'C4'),
                ('C4', 'P4'), 
                ('P4', 'O2')
            ]
    elif args.data in ["BCICIV-2A", "Sleep-EDFx", "BCICIV-2A-TTS", "BCICIV-2A-SC-TTS"]:
        montage = mne.channels.make_standard_montage('standard_1005')
        positions = montage.get_positions()['ch_pos']
        
        if args.data in ["BCICIV-2A", "BCICIV-2A-TTS"]:
            double_banana_pairs = [
                ('C5', 'C1'), 
                ('C3', 'Cz'),
                ('Cz', 'C4'), 
                ('C2', 'C6'),
                ('Fz', 'C5'), 
                ('C5', 'P1'),
                ('CP3', 'POz'), 
                ('Fz', 'C1'),
                ('C1', 'Pz'),
                ('CP1', 'POz'),
                ('Fz', 'C2'), 
                ('C2', 'Pz'),
                ('CP2', 'POz'),
                ('Fz', 'C6'),
                ('C6', 'P2'), 
                ('CP4', 'POz'),
            ]
        elif args.data in ["Sleep-EDFx"]:
            double_banana_pairs = [
                ('Fpz', 'Cz'), 
                ('Pz', 'Oz'),
            ]
        elif args.data in ["BCICIV-2A-SC-TTS"]:
            double_banana_pairs = [
                "Fz",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "P1",
                "Pz",
                "P2",
                "POz",
            ]
    else:
        raise ValueError(f"Do not have the corresponding 3d_pos for {args.data}")
    
    edge_pos_list = []

    for pair in double_banana_pairs:
        if args.data in ["SEED-V", "BCICIV-2B-SC", "BCICIV-2B-SC-TTS", "BCICIV-2A-SC-TTS"]:
            edge_pos = positions[pair]
        else:
            ch1, ch2 = pair
            pos1 = positions[ch1]
            pos2 = positions[ch2]
            edge_pos = (pos1 + pos2) / 2
        
        edge_pos_list.append(edge_pos)
        
    edge_pos_list = np.array(edge_pos_list)
    min_coord = np.min(edge_pos_list, axis=0)
    edge_pos_list = edge_pos_list - min_coord

    scale_factor = max_range / np.max(edge_pos_list)
    edge_pos_list = edge_pos_list * scale_factor
    edge_pos_list = np.rint(edge_pos_list).astype(int)
    
    return edge_pos_list  # size = [dims, x_y_z]


def seperate_3d_dims(d_model):
    if d_model // 3 % 2 == 0:
        x = y = d_model // 3
    else:
        x = y = d_model // 3 - 1    
    z = d_model - x - y
    
    return x, y, z
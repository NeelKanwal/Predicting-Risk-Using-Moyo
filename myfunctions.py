# Author: Neel Kanwal, neel.kanwal0@gmail.com
# This script contains all sort of helper functions, used in training, valuation, plotting and preprocessing. 

import matplotlib.pyplot as plt
import random

import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import math
import seaborn as sns
from collections import Counter
from scipy import linalg
from sklearn.manifold import TSNE
from torch.distributions import Normal, MultivariateNormal
import scipy.io as sio
from torch.nn import LayerNorm
from skimage.metrics import structural_similarity as ssim
import math
import torch.fft as fft


def calculate_fid(real_features, generated_features):
    # Reshape if necessary
    if real_features.ndim > 2:
        real_features = real_features.reshape(real_features.shape[0], -1)
    if generated_features.ndim > 2:
        generated_features = generated_features.reshape(generated_features.shape[0], -1)
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_class_weights(dataset, task_idx):
    class_counts = Counter()
    for _, labels in dataset:
        class_counts.update([label.item() for label in labels[:, task_idx]])

    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Normalize the weights so that they sum to 1
    total_weight = sum(class_weights.values())
    normalized_class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}
    
    return normalized_class_weights

def get_class_weights_0(dataset, task_idx):
    class_counts = Counter()
    for _, labels in dataset:
        class_counts.update([label.item() for label in labels[:, task_idx]])

    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    return class_weights

def weights_to_tensor(class_weights, num_classes):
    weights = torch.ones(num_classes)
    for cls, weight in class_weights.items():
        weights[cls] = weight
    return weights


class FHRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        time_series = npz_data['time_series']

        # # Impute NaN values with the mean of non-NaN values
        # non_nan_values = time_series[~np.isnan(time_series)]
        
        # if len(non_nan_values) > 0:
        #     impute_value = np.mean(non_nan_values)
        # else:
        #     impute_value = 0  # or any other appropriate value

        # time_series = np.where(np.isnan(time_series), impute_value, time_series)
        time_series = np.where(np.isnan(time_series), 0, time_series)

        # Drop NaN values from the beginning and end of the time series
        # time_series = time_series[~np.isnan(time_series)]

        outcome_24hours = npz_data['labels'][-1]
        if outcome_24hours >= 3:
           outcome_24hours = 3
        outcome_24hours = outcome_24hours -1

        outcome_30min = npz_data['labels'][-2]
        if outcome_30min >= 3:
           outcome_30min = 3
        outcome_30min = outcome_30min - 1 
        
        outcome_resus = npz_data['labels'][-6]
        if outcome_resus == 2:
           outcome_resus = 0
        else:
            outcome_resus = 1

        # Add the sequence_length dimension
        # time_series = time_series.unsqueeze(0)  # (sequence_length, input_size) -> (1, sequence_length, input_size)

        return torch.from_numpy(time_series).float(), torch.tensor([outcome_30min, outcome_24hours, outcome_resus]).long()


class FHRDataset_v3(Dataset):
    def __init__(self, data_dir, sequence_length=7200, normalization='minmax'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length
        self.normalization = normalization

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        time_series = npz_data['time_series']

        # Impute NaN values with zeros
        time_series = np.where(np.isnan(time_series), 0, time_series)
        # print("Length of time_series (before):", len(time_series))

        if self.normalization == 'minmax':
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            if max_val > min_val:
                time_series = (time_series - min_val) / (max_val - min_val)
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization == 'zscore':
            mean = np.mean(time_series)
            std = np.std(time_series)
            if std > 0:
                time_series = (time_series - mean) / std
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization is None:
            pass

        ## %% order changed here .... Normalize first and then do padding.
         # If the signal is longer, keep the last part
        if self.sequence_length is not None:
            if time_series.shape[0] > self.sequence_length:
                time_series = time_series[-self.sequence_length:]
            # If the signal is shorter, add zeros at the beginning
            elif time_series.shape[0] < self.sequence_length:
                padding_length = self.sequence_length - time_series.shape[0]
                padding = np.zeros((padding_length,) + time_series.shape[1:])
                time_series = np.concatenate((padding, time_series), axis=0)
        else: 
            pass

        outcome_24hours = npz_data['labels'][-1]
        if outcome_24hours >= 3:
            outcome_24hours = 3
        outcome_24hours -= 1 # start label from zero

        outcome_30min = npz_data['labels'][-2]
        if outcome_30min >= 3:
            outcome_30min = 3
        outcome_30min -= 1 # start label from zero

        outcome_resus = npz_data['labels'][-6]
        if outcome_resus == 2:
            outcome_resus = 0
        else:
            outcome_resus = 1
        time_series = time_series.reshape(-1) # added for VAE (7200) (7200,)

        if self.normalization == 'minmax':   
             # Create min and max vectors
            second_vec = np.full(self.sequence_length, min_val)
            third_vec = np.full(self.sequence_length, max_val)
        elif self.normalization == 'zscore':
            second_vec = np.full(self.sequence_length, mean)
            third_vec = np.full(self.sequence_length, std)
        else:
            raise Exception("Normalization is not used")


        # Reshape arrays if necessary
        time_series = time_series.reshape(-1)
        second_vec = second_vec.reshape(-1)
        third_vec = third_vec.reshape(-1)
        
        # Ensure all arrays have the same length
        min_length = min(len(time_series), len(second_vec), len(third_vec))
        time_series = time_series[:min_length]
        second_vec = second_vec[:min_length]
        third_vec = third_vec[:min_length]
        
        # Now stack the arrays
        combined_series = np.stack([time_series, second_vec, third_vec], axis=-1)

        return torch.from_numpy(time_series).float(),  torch.tensor([outcome_30min, outcome_24hours, outcome_resus]).long()

class FHRDataset_encoder(Dataset):
    def __init__(self, data_dir, sequence_length=7200, normalization='minmax', force_norm=True):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.force_norm = force_norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        time_series = npz_data['time_series']

        # Impute NaN values with zeros
        time_series = np.where(np.isnan(time_series), 0, time_series)

        if self.normalization == 'minmax':
            if self.force_norm: # Fix this for entire dataset normalized in same way
                min_val= 0
                max_val=200
            else:
                min_val = np.min(time_series)
                max_val = np.max(time_series)
            if max_val > min_val:
                time_series = (time_series - min_val) / (max_val - min_val)
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization == 'zscore':
            mean = np.mean(time_series)
            std = np.std(time_series)
            if self.force_norm: # Fix this for entire dataset normalized in same way
                mean = 0
                std = 1
            if std > 0:
                time_series = (time_series - mean) / std
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization is None:
            pass
        # else:
        #     raise Exception("Normalization is not available")

        if self.sequence_length is not None:
            current_length = time_series.shape[0]
            if current_length > self.sequence_length:
                # If the signal is longer, keep the last part
                time_series = time_series[-self.sequence_length:]
            elif current_length < self.sequence_length:
                # If the signal is shorter, add zeros at the beginning
                padding_length = self.sequence_length - current_length
                padding = np.zeros((padding_length,) + time_series.shape[1:])
                time_series = np.concatenate((padding, time_series), axis=0)

            # Double-check to ensure the final length is correct
            assert time_series.shape[0] == self.sequence_length, f"Expected length {self.sequence_length}, got {time_series.shape[0]}"

        outcome_30min, outcome_24hours, outcome_resus = 0, 0, 0

        return torch.from_numpy(time_series).float(),  torch.tensor([outcome_30min, outcome_24hours, outcome_resus]).long()

class FHRDataset_encoder_envelope(Dataset):
    def __init__(self, data_dir, sequence_length=7200, normalization='minmax', force_norm=True):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.force_norm = force_norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        time_series = npz_data['time_series']
        peak_envelope = npz_data['peak_env']
        savgol_envelope = npz_data['savgol_env']

        print(f"File: {self.files[idx]}")
        print(f"Original shapes: time_series {time_series.shape}, peak_envelope {peak_envelope.shape}, savgol_envelope {savgol_envelope.shape}")

        # Ensure all arrays have the same length
        min_length = min(len(time_series), len(peak_envelope), len(savgol_envelope))
        time_series = time_series[:min_length]
        peak_envelope = peak_envelope[:min_length]
        savgol_envelope = savgol_envelope[:min_length]

        print(f"After length adjustment: time_series {time_series.shape}, peak_envelope {peak_envelope.shape}, savgol_envelope {savgol_envelope.shape}")

        # Impute NaN values with zeros
        time_series = np.where(np.isnan(time_series), 0, time_series)
        peak_envelope = np.where(np.isnan(peak_envelope), 0, peak_envelope)
        savgol_envelope = np.where(np.isnan(savgol_envelope), 0, savgol_envelope)

        if self.normalization == 'minmax':
            if self.force_norm: # Fix this for entire dataset normalized in same way
                min_val, max_val= 0,200
            else:
                min_val = np.min(time_series)
                max_val = np.max(time_series)
            if max_val > min_val:
                time_series = (time_series - min_val) / (max_val - min_val)
                peak_envelope = (peak_envelope - min_val) / (max_val - min_val)
                savgol_envelope = (savgol_envelope - min_val) / (max_val - min_val)
            else:
                time_series = np.zeros_like(time_series)
                peak_envelope = np.zeros_like(peak_envelope)
                savgol_envelope = np.zeros_like(savgol_envelope)
        elif self.normalization == 'zscore':
            mean = np.mean(time_series)
            std = np.std(time_series)
            if self.force_norm: # Fix this for entire dataset normalized in same way
                mean = 0
                std = 1
            if std > 0:
                time_series = (time_series - mean) / std
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization is None:
            pass
        # else:
        #     raise Exception("Normalization is not available")

        # Adjust sequence length for all three signals
        if self.sequence_length is not None:
            current_length = time_series.shape[0]
            if current_length > self.sequence_length:
                time_series = time_series[-self.sequence_length:]
                peak_envelope = peak_envelope[-self.sequence_length:]
                savgol_envelope = savgol_envelope[-self.sequence_length:]
            elif current_length < self.sequence_length:
                padding_length = self.sequence_length - current_length
                padding = np.zeros(padding_length)
                time_series = np.concatenate((padding, time_series))
                peak_envelope = np.concatenate((padding, peak_envelope))
                savgol_envelope = np.concatenate((padding, savgol_envelope))

            assert time_series.shape[0] == self.sequence_length, f"Expected length {self.sequence_length}, got {time_series.shape[0]}"

        # Stack the three signals as channels
        combined_signal = np.stack([time_series, peak_envelope, savgol_envelope], axis=0)

        outcome_30min, outcome_24hours, outcome_resus = 0, 0, 0

        return torch.from_numpy(combined_signal).float(), torch.tensor([outcome_30min, outcome_24hours, outcome_resus]).long()

class FHRDataset_encoder_v2(Dataset):
    # this make min and max vector and supplies along with the original signal.
    def __init__(self, data_dir, sequence_length=7200, normalization='minmax'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length
        self.normalization = normalization

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        time_series = npz_data['time_series']

        # Impute NaN values with zeros
        time_series = np.where(np.isnan(time_series), 0, time_series)

        if self.normalization == 'minmax':
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            if max_val > min_val:
                time_series = (time_series - min_val) / (max_val - min_val)
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization == 'zscore':
            mean = np.mean(time_series)
            std = np.std(time_series)
            if std > 0:
                time_series = (time_series - mean) / std
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization is None:
            pass
        else:
            raise Exception("Normalization is not available")

        if self.sequence_length is not None:
            current_length = time_series.shape[0]
            if current_length > self.sequence_length:
                # If the signal is longer, keep the last part
                time_series = time_series[-self.sequence_length:]
            elif current_length < self.sequence_length:
                # If the signal is shorter, add zeros at the beginning
                padding_length = self.sequence_length - current_length
                padding = np.zeros((padding_length,) + time_series.shape[1:])
                time_series = np.concatenate((padding, time_series), axis=0)

            # Double-check to ensure the final length is correct
            assert time_series.shape[0] == self.sequence_length, f"Expected length {self.sequence_length}, got {time_series.shape[0]}"


        if self.normalization == 'minmax':   
             # Create min and max vectors
            second_vec = np.full(self.sequence_length, min_val)
            third_vec = np.full(self.sequence_length, max_val)
        elif self.normalization == 'zscore':
            second_vec = np.full(self.sequence_length, mean)
            third_vec = np.full(self.sequence_length, std)
        else:
            raise Exception("Normalization is not used")

        # Reshape arrays if necessary
        time_series = time_series.reshape(-1)
        second_vec = second_vec.reshape(-1)
        third_vec = third_vec.reshape(-1)
        
        # Ensure all arrays have the same length
        min_length = min(len(time_series), len(second_vec), len(third_vec))
        time_series = time_series[:min_length]
        second_vec = second_vec[:min_length]
        third_vec = third_vec[:min_length]
        
        # Now stack the arrays
        combined_series = np.stack([time_series, second_vec, third_vec], axis=-1)
        # print(combined_series.shape)

        outcome_30min, outcome_24hours, outcome_resus = 0, 0, 0

        return torch.from_numpy(combined_series).float(),  torch.tensor([outcome_30min, outcome_24hours, outcome_resus]).long()


def plot_random_signals_old(data_loader, save_path, norm, num_signals=5, figsize=(30, 20)):
   # Collect enough samples
    print("Plotting Signals from the training")
    all_data = []
    all_files = []
    while len(all_data) < num_signals:
        data, _ = next(iter(data_loader))
        all_data.extend(data)
        all_files.extend([data_loader.dataset.files[i] for i in range(len(data))])
        
    # Create a list of random indices
    num_signals = min(num_signals, len(all_data))
    # random_indices = random.sample(range(len(all_data)), num_signals)
    # Instead of random try first five to observe the change
    random_indices = [1,2,3,4,5]
    
    # Create the plot
    fig, axes = plt.subplots(num_signals, 1, figsize=figsize)
    fig.suptitle(f"Random Signals from Training Dataset with {norm}", fontsize=32)
    
    if num_signals == 1:
        axes = [axes]

    for i, idx in enumerate(random_indices):
        signal = all_data[idx].numpy()

        # Plot the time series
        axes[i].plot(signal, label='Nomralized Time Series', alpha=0.8)

         # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)

        # # Plot the signal
        # axes[i].plot(signal[0])
     
        # Get the file name
        file_name = os.path.basename(all_files[idx])
        
        # Set the title of the subplot
        axes[i].set_title(f"{file_name},     Min: {min_value:.2f},    Max: {max_value:.2f}", fontsize=28)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
        # axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/random_signals_from_training.png')
    plt.close()


def plot_random_signals(data_loader,  model_name, save_path, norm, num_signals=3, figsize=(30, 20)):
   # Collect enough samples
    print(f"Plotting Signals from the training using {norm}.")
    all_data = []
    all_files = []
    while len(all_data) < num_signals:
        data, _ = next(iter(data_loader))
        all_data.extend(data)
        all_files.extend([data_loader.dataset.files[i] for i in range(len(data))])
        
    # Create a list of random indices
    num_signals = min(num_signals, len(all_data))
    # random_indices = list(range(0,num_signals))
    random_indices = list(range(num_signals))
    
    # Create the plot
    fig, axes = plt.subplots(num_signals, 1, figsize=figsize)
    fig.suptitle(f"Random Signals from Training Dataset with {norm} and {model_name}", fontsize=32)
    
    if num_signals == 1:
        axes = [axes]

    for i, idx in enumerate(random_indices):
        signal = all_data[idx].numpy()

        # Plot the time series
        axes[i].plot(signal, label='Nomralized Time Series', alpha=0.9)

        axes[i].grid(True, linestyle='--', alpha=0.7)

        file_name = os.path.basename(all_files[idx])
        
        # Set the title of the subplot
        axes[i].set_title(f"{file_name}", fontsize=28)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/random_signals_from_training.png')
    plt.close()


def plot_random_signals_v2(data_loader, model_name, save_path, norm, num_signals=3, figsize=(30, 20)):
    # Collect enough samples
    print(f"Plotting Signals from the training using {norm}.")
    all_data = []
    all_files = []

    # Iterate over the data loader to collect samples
    for data,  _, _ in data_loader:
        all_data.extend(data)
        all_files.extend([data_loader.dataset.files[i] for i in range(len(data))])
        if len(all_data) >= num_signals:
            break

    # Create a list of random indices
    num_signals = min(num_signals, len(all_data))
    random_indices = list(range(num_signals))

    # Create the plot
    fig, axes = plt.subplots(num_signals, 1, figsize=figsize)
    fig.suptitle(f"Random Signals from Training Dataset with {norm} and {model_name}", fontsize=32)

    if num_signals == 1:
        axes = [axes]

    for i, idx in enumerate(random_indices):
        signal = all_data[idx].numpy()

        # Plot the time series
        axes[i].plot(signal, label='Normalized Time Series', alpha=0.9)

        axes[i].grid(True, linestyle='--', alpha=0.7)

        file_name = os.path.basename(all_files[idx])

        # Set the title of the subplot
        axes[i].set_title(f"{file_name}", fontsize=28)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(f'{save_path}/random_signals_from_training.png')
    plt.close()

def plot_random_signals_v3(data_loader, model_name, save_path, norm, num_signals=3, figsize=(30, 20)):
    # for stacked signals when signals are two channel input_dim
    print(f"Plotting Signals from the training using {norm}.")
    all_data = []
    all_files = []

    # Iterate over the data loader to collect samples
    for data, _, _ in data_loader:
        all_data.extend(data)
        all_files.extend([data_loader.dataset.files[i] for i in range(len(data))])
        if len(all_data) >= num_signals:
            break

    # Create a list of random indices
    num_signals = min(num_signals, len(all_data))
    random_indices = np.random.choice(len(all_data), num_signals, replace=False)

    # Create the plot
    fig, axes = plt.subplots(num_signals, 1, figsize=figsize)
    fig.suptitle(f"Random Signals from Training Dataset with {norm} and {model_name}", fontsize=32)

    if num_signals == 1:
        axes = [axes]

    for i, idx in enumerate(random_indices):
        signals = all_data[idx].numpy()  # Shape: (2, sequence_length)
        signal_hybrid60 = signals[0]
        peak_envelope = signals[1]

        # Plot both signals
        axes[i].plot(signal_hybrid60, label='Signal Hybrid60', alpha=0.9)
        axes[i].plot(peak_envelope, label='Peak Envelope', alpha=0.9)

        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend(fontsize=12)

        file_name = os.path.basename(all_files[idx])

        # Set the title of the subplot
        axes[i].set_title(f"{file_name}", fontsize=28)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(f'{save_path}/random_signals_from_training.png')
    plt.close()

def plot_random_signals_with_envelope(data_loader, model_name, save_path, norm, num_signals=3, figsize=(30, 20)):
    print(f"Plotting Signals from the training using {norm}.")
    all_data = []
    all_files = []
    while len(all_data) < num_signals:
        data, _ = next(iter(data_loader))
        all_data.extend(data)
        all_files.extend([data_loader.dataset.files[i] for i in range(len(data))])
        
    num_signals = min(num_signals, len(all_data))
    random_indices = list(range(num_signals))
    
    fig, axes = plt.subplots(num_signals, 1, figsize=figsize)
    fig.suptitle(f"Random Signals from Training Dataset with {norm} and {model_name}", fontsize=32)
    
    if num_signals == 1:
        axes = [axes]

    for i, idx in enumerate(random_indices):
        signal = all_data[idx].numpy()  # shape: (3, sequence_length)
        
        time_series = signal[0]
        peak_envelope = signal[1]
        savgol_envelope = signal[2]

        # Plot all three channels
        axes[i].plot(time_series, label='Normalized (Imputed) Time Series', alpha=0.9, color='blue')
        axes[i].plot(peak_envelope, label='Peak Envelope', alpha=0.7, color='red')
        axes[i].plot(savgol_envelope, label='Savgol Envelope', alpha=0.7, color='green')

        axes[i].grid(True, linestyle='--', alpha=0.7)

        file_name = os.path.basename(all_files[idx])
        
        axes[i].set_title(f"{file_name}", fontsize=28)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/random_signals_from_training.png')
    plt.close()

class VAE(nn.Module):
    def __init__(self, input_dim=7200, hidden_dim=1024, latent_dim=256):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),  # Changed from hidden_dim to hidden_dim//2
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU())
        
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),  # Changed from hidden_dim to hidden_dim//2
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),  # Changed from hidden_dim to hidden_dim//2
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        # print("Encode input shape:", x.shape)
        h = self.encoder(x)
        # print("Encoder output shape:", h.shape)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = x.squeeze(-1)  # Remove the last dimension if it's 1
        # print("Shape after squeeze:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten the input
        # print("Shape after flatten:", x.shape)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z).unsqueeze(-1), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=0.9):
    x = x.squeeze(-1)  # Remove the last dimension if it's 1
    recon_x = recon_x.squeeze(-1)  # Remove the last dimension if it's 1
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


class VAE_v2(nn.Module):
    def __init__(self, input_dim=7200, hidden_dim=1024, latent_dim=256):
        super(VAE_v2, self).__init__()
        self.input_dim = input_dim
        self.input_dim_3d = input_dim * 3

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim_3d, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU())
        
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim_3d))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        original_shape = x.shape
        
        if len(x.shape) == 3:
            x = x.view(x.size(0), -1)  # Flatten (batch_size, 7200, 3) to (batch_size, 21600)
        elif len(x.shape) == 2:
            x = x.repeat(1, 3)  # Repeat (batch_size, 7200) to (batch_size, 21600)
        
        # print(f"Processed shape: {x.shape}")
        # print(f"Expected encoder input shape: {self.input_dim_3d}")
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        
        # Reshape the output based on the original input shape
        if len(original_shape) == 3:
            reconstructed = reconstructed.view(original_shape)
        else:
            reconstructed = reconstructed.view(original_shape[0], -1)[:, :self.input_dim]
        
        # print(f"Reconstructed shape: {reconstructed.shape}")
        return reconstructed, mu, logvar

def vae_loss_v2(recon_x, x, mu, logvar, beta=0.9):
    # print(f"recon_x shape: {recon_x.shape}, x shape: {x.shape}")
    
    if len(x.shape) == 3:
        x_flat = x.view(x.size(0), -1)
        recon_x_flat = recon_x.view(recon_x.size(0), -1)
    else:
        x_flat = x
        recon_x_flat = recon_x
    
    # print(f"Flattened shapes - recon_x: {recon_x_flat.shape}, x: {x_flat.shape}")
    
    assert x_flat.shape == recon_x_flat.shape, f"Shapes do not match: {x_flat.shape} vs {recon_x_flat.shape}"
    
    MSE = F.mse_loss(recon_x_flat, x_flat, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

def import_mat(filename):
    # Import data from *.mat file
    imported_mat = sio.loadmat(filename)
    acc_enrg = imported_mat['moyoData']['acc_enrg'][0][0]
    acc_x_mg = imported_mat['moyoData']['acc_x_mg'][0][0]
    acc_y_mg = imported_mat['moyoData']['acc_y_mg'][0][0]
    acc_z_mg = imported_mat['moyoData']['acc_z_mg'][0][0]
    fhr = imported_mat['moyoData']['fhr'][0][0]
    fhr_qual = imported_mat['moyoData']['fhr_qual'][0][0]
    outcome = imported_mat['moyoData']['outcome'][0][0][0][0]
    cleaninfo_cleanFHR = imported_mat['moyoData']['cleanInfo'][0][0][0]['cleanFHR'][0]
    cleaninfo_interpolated = imported_mat['moyoData']['cleanInfo'][0][0][0]['masks'][0][0][0]['interpolated']
    
    return acc_enrg, acc_x_mg, acc_y_mg, acc_z_mg, fhr, fhr_qual, outcome, cleaninfo_cleanFHR, cleaninfo_interpolated

class FHRDataset_v2(Dataset):
    def __init__(self, data_dir, sequence_length=7200, normalization='minmax', force_norm=True):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length
        self.normalization = normalization
        self.force_norm = force_norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        time_series = npz_data['time_series']

        # Impute NaN values with zeros
        time_series = np.where(np.isnan(time_series), 0, time_series)
        # print("Length of time_series (before):", len(time_series))

        if self.normalization == 'minmax':
            if self.force_norm: # Fix this for entire dataset normalized in same way
                min_val= 0
                max_val=200
            else:
                min_val = np.min(time_series)
                max_val = np.max(time_series)
            if max_val > min_val:
                time_series = (time_series - min_val) / (max_val - min_val + 1e-8)
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization == 'zscore':
            mean = np.mean(time_series)
            std = np.std(time_series)
            if self.force_norm: # Fix this for entire dataset normalized in same way
                mean = 0
                std = 1
            if std > 0:
                time_series = (time_series - mean) / std
            else:
                time_series = np.zeros_like(time_series)
        elif self.normalization is None:
            pass

        ## %% order changed here .... Normalize first and then do padding.
         # If the signal is longer, keep the last part
        if self.sequence_length is not None:
            if time_series.shape[0] > self.sequence_length:
                time_series = time_series[-self.sequence_length:]
            # If the signal is shorter, add zeros at the beginning
            elif time_series.shape[0] < self.sequence_length:
                padding_length = self.sequence_length - time_series.shape[0]
                padding = np.zeros((padding_length,) + time_series.shape[1:])
                time_series = np.concatenate((padding, time_series), axis=0)
        else: 
            pass

        outcome_24hours = npz_data['labels'][-1]
        if outcome_24hours >= 3:
            outcome_24hours = 3
        outcome_24hours -= 1

        outcome_30min = npz_data['labels'][-2]
        if outcome_30min >= 3:
            outcome_30min = 3
        outcome_30min -= 1 

        outcome_resus = npz_data['labels'][-6]
        if outcome_resus == 2:
            outcome_resus = 0 # NOT ATTEMPTED
        else:
            outcome_resus = 1 # ATTEMPTED

        time_series = time_series.reshape(self.sequence_length, 1)
        # time_series = time_series.reshape(-1) # added for VAE

        return torch.from_numpy(time_series).float(),  torch.tensor([outcome_30min, outcome_24hours, outcome_resus]).long()


class FHRDataset_v2_envelope(Dataset):
    def __init__(self, data_dir, sequence_length=7200, normalization='minmax'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length
        self.normalization = normalization

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        time_series = npz_data['time_series']
        interp_mask = npz_data['interp_mask']
        fhr_original = npz_data['fhr_original']
        peak_envelope = npz_data['peak_env']
        savgol_envelope = npz_data['savgol_env']

        # print(f"File: {self.files[idx]}")
        # print(f"Original shapes:")
        # print(f"  time_series: {time_series.shape}")
        # print(f"  fhr_original: {fhr_original.shape}")
        # print(f"  peak_envelope: {peak_envelope.shape}")
        # print(f"  savgol_envelope: {savgol_envelope.shape}")

        # Ensure all arrays are 1-dimensional
        time_series = time_series.flatten()
        fhr_original = fhr_original.flatten()
        interp_mask = interp_mask.flatten()
        peak_envelope = peak_envelope.flatten()
        savgol_envelope = savgol_envelope.flatten()

        # Ensure all arrays have the same length
        min_length = min(len(time_series), len(peak_envelope), len(savgol_envelope), len(fhr_original))
        time_series = time_series[:min_length]
        fhr_original = fhr_original[:min_length]
        interp_mask = interp_mask[:min_length]
        peak_envelope = peak_envelope[:min_length]
        savgol_envelope = savgol_envelope[:min_length]

        # print(f"After flattening and truncation to min length ({min_length}):")
        # print(f"  time_series: {time_series.shape}")
        # print(f"  fhr_original: {fhr_original.shape}")
        # print(f"  peak_envelope: {peak_envelope.shape}")
        # print(f"  savgol_envelope: {savgol_envelope.shape}")

        # Impute NaN values with zeros for all signals
        time_series = np.nan_to_num(time_series, nan=0.0)
        fhr_original = np.nan_to_num(fhr_original, nan=0.0)
        interp_mask = np.nan_to_num(interp_mask, nan=0.0)
        peak_envelope = np.nan_to_num(peak_envelope, nan=0.0)
        savgol_envelope = np.nan_to_num(savgol_envelope, nan=0.0)

        # Normalize all four signals
        if self.normalization == 'minmax':
            min_val, max_val = 0, 220 
            if max_val > min_val:
                time_series = (time_series - min_val) / (max_val - min_val + 1e-8)
                fhr_original = (fhr_original - min_val) / (max_val - min_val + 1e-8)
                peak_envelope = (peak_envelope - min_val) / (max_val - min_val + 1e-8)
                savgol_envelope = (savgol_envelope - min_val) / (max_val - min_val + 1e-8)
            else:
                time_series = np.zeros_like(time_series)
                fhr_original = np.zeros_like(fhr_original)
                peak_envelope = np.zeros_like(peak_envelope)
                savgol_envelope = np.zeros_like(savgol_envelope)
        elif self.normalization is None:
            pass

        if self.sequence_length is not None:
            if len(time_series) > self.sequence_length:
                # Always select the last self.sequence_length samples
                time_series = time_series[-self.sequence_length:]
                fhr_original = fhr_original[-self.sequence_length:]
                interp_mask = interp_mask[-self.sequence_length:]
                peak_envelope = peak_envelope[-self.sequence_length:]
                savgol_envelope = savgol_envelope[-self.sequence_length:]
               
            elif len(time_series) < self.sequence_length:
                padding_length = self.sequence_length - len(time_series)
                # Pad at the beginning of the signal
                time_series = np.pad(time_series, (padding_length, 0), mode='constant', constant_values=0)
                fhr_original = np.pad(fhr_original, (padding_length, 0), mode='constant', constant_values=0)
                interp_mask = np.pad(interp_mask, (padding_length, 0), mode='constant', constant_values=0)
                peak_envelope = np.pad(peak_envelope, (padding_length, 0), mode='constant', constant_values=0)
                savgol_envelope = np.pad(savgol_envelope, (padding_length, 0), mode='constant', constant_values=0)
               

        # print(f"Final shapes after sequence length adjustment:")
        # print(f"  time_series: {time_series.shape}")
        # print(f"  fhr_original: {fhr_original.shape}")
        # print(f"  peak_envelope: {peak_envelope.shape}")
        # print(f"  savgol_envelope: {savgol_envelope.shape}")

        # Stack the signals as channels
        combined_signal = np.stack([time_series, peak_envelope, savgol_envelope], axis=0)

        # print(f"Shape of combined_signal: {combined_signal.shape}")

        return (
            torch.from_numpy(combined_signal).float(),
            torch.from_numpy(fhr_original).float(),
            torch.from_numpy(interp_mask).float(),
            torch.tensor([0, 0, 0]).long())

class FHRDataset_v3_envelope(Dataset):
    def __init__(self, data_dir, sequence_length=7200, normalization='minmax'):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length
        self.normalization = normalization

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        fhr = npz_data['inpainted_hybrid_30']
        peak_envelope = npz_data['peak_env']
        savgol_envelope = npz_data['savgol_env']

        # Ensure all arrays are 1-dimensional
        fhr_original = fhr.flatten()
        peak_envelope = peak_envelope.flatten()
        savgol_envelope = savgol_envelope.flatten()

        # Ensure all arrays have the same length
        min_length = min(len(peak_envelope), len(savgol_envelope), len(fhr))
        fhr = fhr[:min_length]
        peak_envelope = peak_envelope[:min_length]
        savgol_envelope = savgol_envelope[:min_length]

        fhr = np.nan_to_num(fhr, nan=0.0)
        peak_envelope = np.nan_to_num(peak_envelope, nan=0.0)
        savgol_envelope = np.nan_to_num(savgol_envelope, nan=0.0)

        # Normalize all four signals
        if self.normalization == 'minmax':
            min_val, max_val = 0, 200 
            if max_val > min_val:
                fhr = (fhr - min_val) / (max_val - min_val + 1e-8)
                peak_envelope = (peak_envelope - min_val) / (max_val - min_val + 1e-8)
                savgol_envelope = (savgol_envelope - min_val) / (max_val - min_val + 1e-8)
            else:
                fhr = np.zeros_like(fhr)
                peak_envelope = np.zeros_like(peak_envelope)
                savgol_envelope = np.zeros_like(savgol_envelope)
        elif self.normalization is None:
            pass

        if self.sequence_length is not None:
            if len(time_series) > self.sequence_length:
                fhr = fhr[-self.sequence_length:]
                peak_envelope = peak_envelope[-self.sequence_length:]
                savgol_envelope = savgol_envelope[-self.sequence_length:]
               
            elif len(fhr) < self.sequence_length:
                padding_length = self.sequence_length - len(fhr)
                # Pad at the beginning of the signal
                fhr = np.pad(fhr, (padding_length, 0), mode='constant', constant_values=0)
                peak_envelope = np.pad(peak_envelope, (padding_length, 0), mode='constant', constant_values=0)
                savgol_envelope = np.pad(savgol_envelope, (padding_length, 0), mode='constant', constant_values=0)
               
        # Stack the signals as channels
        combined_signal = np.stack([fhr, peak_envelope, savgol_envelope], axis=0)

        return (torch.from_numpy(combined_signal).float(),torch.tensor([0, 0, 0]).long())



class FHRDataset_v4_envelope(Dataset):
    # USED in explore_new_dataset.py file to fetch new files for visualization
    def __init__(self, data_dir, sequence_length=7200):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        fhr_original = npz_data['fhr_original_normalized']
        time_series = npz_data['time_series_normalized']
        interp_mask = npz_data['interp_mask']
        peak_envelope = npz_data['peak_env']
        savgol_envelope = npz_data['savgol_env']
        signal_hybrid30 = npz_data['savgol_env']
        signal_hybrid60 = npz_data['savgol_env']
        # signal_hybrid30 = npz_data['inpainted_model_2']
        # signal_hybrid60 = npz_data['inpainted_model_1'] # better with fluctuations



        # Ensure all arrays are 1-dimensional
        time_series = time_series.flatten()
        fhr_original = fhr_original.flatten()
        interp_mask = interp_mask.flatten()
        peak_envelope = peak_envelope.flatten()
        savgol_envelope = savgol_envelope.flatten()

        signal_hybrid30 = signal_hybrid30.flatten()
        signal_hybrid60 = signal_hybrid60.flatten()

        # Ensure all arrays have the same length
        min_length = min(len(time_series), len(peak_envelope), len(savgol_envelope), len(fhr_original))
        time_series = time_series[:min_length]
        fhr_original = fhr_original[:min_length]
        interp_mask = interp_mask[:min_length]
        peak_envelope = peak_envelope[:min_length]
        savgol_envelope = savgol_envelope[:min_length]
        signal_hybrid30 = signal_hybrid30[:min_length]
        signal_hybrid60 = signal_hybrid60[:min_length]


        # Impute NaN values with zeros for all signals
        time_series = np.nan_to_num(time_series, nan=0.0)
        fhr_original = np.nan_to_num(fhr_original, nan=0.0)
        interp_mask = np.nan_to_num(interp_mask, nan=0.0)
        peak_envelope = np.nan_to_num(peak_envelope, nan=0.0)
        savgol_envelope = np.nan_to_num(savgol_envelope, nan=0.0)
        signal_hybrid30 = np.nan_to_num(signal_hybrid30, nan=0.0)
        signal_hybrid60 = np.nan_to_num(signal_hybrid60, nan=0.0)


        if self.sequence_length is not None:
            if len(time_series) > self.sequence_length:
                # Always select the last self.sequence_length samples
                time_series = time_series[-self.sequence_length:]
                fhr_original = fhr_original[-self.sequence_length:]
                interp_mask = interp_mask[-self.sequence_length:]
                peak_envelope = peak_envelope[-self.sequence_length:]
                savgol_envelope = savgol_envelope[-self.sequence_length:]
                signal_hybrid30 = signal_hybrid30[-self.sequence_length:]
                signal_hybrid60 = signal_hybrid60[-self.sequence_length:]
               
            elif len(time_series) < self.sequence_length:
                padding_length = self.sequence_length - len(time_series)
                # Pad at the beginning of the signal
                time_series = np.pad(time_series, (padding_length, 0), mode='constant', constant_values=0)
                fhr_original = np.pad(fhr_original, (padding_length, 0), mode='constant', constant_values=0)
                interp_mask = np.pad(interp_mask, (padding_length, 0), mode='constant', constant_values=0)
                peak_envelope = np.pad(peak_envelope, (padding_length, 0), mode='constant', constant_values=0)
                savgol_envelope = np.pad(savgol_envelope, (padding_length, 0), mode='constant', constant_values=0)
                signal_hybrid30 = np.pad(signal_hybrid30, (padding_length, 0), mode='constant', constant_values=0)
                signal_hybrid60 = np.pad(signal_hybrid60, (padding_length, 0), mode='constant', constant_values=0)
               

        return (torch.from_numpy(fhr_original).float(),
                torch.from_numpy(time_series).float(),
                torch.from_numpy(peak_envelope).float(),
                torch.from_numpy(savgol_envelope).float(),
                torch.from_numpy(interp_mask).float(),
                torch.from_numpy(signal_hybrid30).float(),
                torch.from_numpy(signal_hybrid60).float()), torch.tensor([0, 0, 0]).long()

class FHRDataset_v5_envelope(Dataset):
    # USED in explore_new_dataset.py file to fetch new files for visualization
    def __init__(self, data_dir, sequence_length=7200):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        fhr_original = npz_data['fhr_original_normalized']
        time_series = npz_data['time_series_normalized']
        interp_mask = npz_data['interp_mask']
        peak_envelope = npz_data['peak_env']
        savgol_envelope = npz_data['savgol_env']
        # signal_hybrid30 = npz_data['inpainted_hybrid_30']
        # signal_hybrid60 = npz_data['inpainted_hybrid_60']
        signal_hybrid30 = npz_data['inpainted_model_2']
        signal_hybrid60 = npz_data['inpainted_model_1'] # better with fluctuations


        # Ensure all arrays are 1-dimensional
        time_series = time_series.flatten()
        fhr_original = fhr_original.flatten()
        interp_mask = interp_mask.flatten()
        peak_envelope = peak_envelope.flatten()
        savgol_envelope = savgol_envelope.flatten()

        signal_hybrid30 = signal_hybrid30.flatten()
        signal_hybrid60 = signal_hybrid60.flatten()

        # Ensure all arrays have the same length
        min_length = min(len(time_series), len(peak_envelope), len(savgol_envelope), len(fhr_original))
        time_series = time_series[:min_length]
        fhr_original = fhr_original[:min_length]
        interp_mask = interp_mask[:min_length]
        peak_envelope = peak_envelope[:min_length]
        savgol_envelope = savgol_envelope[:min_length]
        signal_hybrid30 = signal_hybrid30[:min_length]
        signal_hybrid60 = signal_hybrid60[:min_length]


        # Impute NaN values with zeros for all signals
        time_series = np.nan_to_num(time_series, nan=0.0)
        fhr_original = np.nan_to_num(fhr_original, nan=0.0)
        interp_mask = np.nan_to_num(interp_mask, nan=0.0)
        peak_envelope = np.nan_to_num(peak_envelope, nan=0.0)
        savgol_envelope = np.nan_to_num(savgol_envelope, nan=0.0)
        signal_hybrid30 = np.nan_to_num(signal_hybrid30, nan=0.0)
        signal_hybrid60 = np.nan_to_num(signal_hybrid60, nan=0.0)


        if self.sequence_length is not None:
            if len(time_series) > self.sequence_length:
                # Always select the last self.sequence_length samples
                time_series = time_series[-self.sequence_length:]
                fhr_original = fhr_original[-self.sequence_length:]
                interp_mask = interp_mask[-self.sequence_length:]
                peak_envelope = peak_envelope[-self.sequence_length:]
                savgol_envelope = savgol_envelope[-self.sequence_length:]
                signal_hybrid30 = signal_hybrid30[-self.sequence_length:]
                signal_hybrid60 = signal_hybrid60[-self.sequence_length:]
               
            elif len(time_series) < self.sequence_length:
                padding_length = self.sequence_length - len(time_series)
                # Pad at the beginning of the signal
                time_series = np.pad(time_series, (padding_length, 0), mode='constant', constant_values=0)
                fhr_original = np.pad(fhr_original, (padding_length, 0), mode='constant', constant_values=0)
                interp_mask = np.pad(interp_mask, (padding_length, 0), mode='constant', constant_values=0)
                peak_envelope = np.pad(peak_envelope, (padding_length, 0), mode='constant', constant_values=0)
                savgol_envelope = np.pad(savgol_envelope, (padding_length, 0), mode='constant', constant_values=0)
                signal_hybrid30 = np.pad(signal_hybrid30, (padding_length, 0), mode='constant', constant_values=0)
                signal_hybrid60 = np.pad(signal_hybrid60, (padding_length, 0), mode='constant', constant_values=0)
               

        # return (torch.from_numpy(fhr_original).float(),
        #         torch.from_numpy(time_series).float(),
        #         torch.from_numpy(peak_envelope).float(),
        #         torch.from_numpy(savgol_envelope).float(),
        #         torch.from_numpy(interp_mask).float(),
        #         torch.from_numpy(signal_hybrid30).float(),
        #         torch.from_numpy(signal_hybrid60).float()), torch.tensor([0, 0, 0]).long()

        return torch.from_numpy(signal_hybrid60).float(), torch.tensor([0, 0, 0]).long()


class FHRDataset_v6(Dataset):
    # USED in training transformer with labelled data using just time series signal/ single channel
    def __init__(self, data_dir, sequence_length=7200):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        # new_data = dict(npz_data)
        # print(f"Available keys in new_data: {new_data.keys()}")
        time_series = npz_data['time_series_normalized']
        signal_hybrid60 = npz_data['inpainted_model_1'] # better with fluctuations
        interp_mask = npz_data['interp_mask']


        # Ensure all arrays are 1-dimensional
        time_series = time_series.flatten()
        signal_hybrid60 = signal_hybrid60.flatten()
        interp_mask = interp_mask.flatten()

        # Ensure all arrays have the same length
        min_length = min(len(time_series), len(signal_hybrid60))
        time_series = time_series[:min_length]
        signal_hybrid60 = signal_hybrid60[:min_length]
        interp_mask = interp_mask[:min_length]


        # Impute NaN values with zeros for all signals
        time_series = np.nan_to_num(time_series, nan=0.0)
        signal_hybrid60 = np.nan_to_num(signal_hybrid60, nan=0.0)
        interp_mask = np.nan_to_num(interp_mask, nan=0.0)

        if self.sequence_length is not None:
            if len(time_series) > self.sequence_length:
                # Always select the last self.sequence_length samples
                time_series = time_series[-self.sequence_length:]
                signal_hybrid60 = signal_hybrid60[-self.sequence_length:]
                interp_mask = interp_mask[-self.sequence_length:]
               
            elif len(time_series) < self.sequence_length:
                padding_length = self.sequence_length - len(time_series)
                time_series = np.pad(time_series, (padding_length, 0), mode='constant', constant_values=0)
                signal_hybrid60 = np.pad(signal_hybrid60, (padding_length, 0), mode='constant', constant_values=0)
                interp_mask = np.pad(interp_mask, (padding_length, 0), mode='constant', constant_values=0)

        # outcome_24hours = npz_data['labels'][-1]
        # if outcome_24hours >= 3:
        #     outcome_24hours = 3
        # outcome_24hours -= 1

        outcome_30min = npz_data['labels'][-2]
        if outcome_30min >= 3:
            outcome_30min = 3
        outcome_30min -= 1 

        # outcome_resus = npz_data['labels'][-6]
        # if outcome_resus == 2:
        #     outcome_resus = 0 # NOT ATTEMPTED
        # else:
        #     outcome_resus = 1 # ATTEMPTED

        outcome_bmv = npz_data['labels'][-3]
        if outcome_bmv == 2:
            outcome_bmv = 0 # NOT ATTEMPTED
        else:
            outcome_bmv = 1 # ATTEMPTED

        time_series = time_series.reshape(self.sequence_length, 1)
        outcome_30min_tensor = torch.tensor(outcome_30min).long()
        outcome_bmv_tensor = torch.tensor(outcome_bmv).long()
        signal_hybrid60 = torch.from_numpy(signal_hybrid60).float()
        interp_mask = torch.from_numpy(interp_mask).float()

        return signal_hybrid60, interp_mask, outcome_30min_tensor, outcome_bmv_tensor


class FHRDataset_v6_synthetic(Dataset):
    def __init__(self, num_samples=100, sequence_length=7200):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        # Precompute class labels with 80% majority (class 0) and 20% minority (class 1)
        self.labels = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the predetermined class label for this sample
        cls_label = self.labels[idx]
        time = np.linspace(0, 1, self.sequence_length)

        if cls_label == 0:
            # Majority class: Sine wave at 5 Hz with moderate noise
            signal = np.sin(2 * np.pi * 5 * time) + np.random.normal(0, 0.1, self.sequence_length)
        else:
            # Minority class: Sine wave at 7 Hz with increased noise and a slight linear trend
            signal = 0.95 *np.sin(2 * np.pi * 7 * time) + np.random.normal(0, 0.2, self.sequence_length)
            signal += np.linspace(0, 0.5, self.sequence_length)

        # Normalize the signal between 0 and 1
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        # Create a random interpolation mask
        interp_mask = np.random.choice([0, 1], size=self.sequence_length, p=[0.1, 0.9])
        # Generate another random outcome for demonstration
        outcome_30min = np.random.randint(0, 3)
        # Use the class label as the binary outcome (0: majority, 1: minority)
        outcome_bmv = cls_label

        # Convert results to tensors
        signal_tensor = torch.from_numpy(signal).float()
        interp_mask_tensor = torch.from_numpy(interp_mask).float()
        outcome_30min_tensor = torch.tensor(outcome_30min).long()
        outcome_bmv_tensor = torch.tensor(outcome_bmv).long()

        return signal_tensor, interp_mask_tensor, outcome_30min_tensor, outcome_bmv_tensor



class FHRDataset_v6_envelope(Dataset):
    # USED in training transformer with labelled data using just time series signal/ single channel
    def __init__(self, data_dir, sequence_length=7200):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz_data = np.load(file_path)
        # new_data = dict(npz_data)
        # print(f"Available keys in new_data: {new_data.keys()}")
        # time_series = npz_data['time_series_normalized']
        signal_hybrid60 = npz_data['inpainted_model_1'] # better with fluctuations
        peak_envelope = npz_data['savgol_env']
        # peak_envelope = npz_data['peak_env']
        # savgol_envelope = npz_data['savgol_env']
       

        # Ensure all arrays are 1-dimensional
        # time_series = time_series.flatten()
        signal_hybrid60 = signal_hybrid60.flatten()
        peak_envelope = peak_envelope.flatten()

        # mean = np.mean(peak_envelope)
        # std = np.std(peak_envelope)
        # peak_envelope = (peak_envelope - mean) / std

        min_val, max_val = 0, 200 
        peak_envelope = (peak_envelope - min_val) / (max_val - min_val + 1e-8)

        # Ensure all arrays have the same length
        min_length = min(len(peak_envelope), len(signal_hybrid60))
        # time_series = time_series[:min_length]
        signal_hybrid60 = signal_hybrid60[:min_length]
        peak_envelope = peak_envelope[:min_length]


        # Impute NaN values with zeros for all signals
        # time_series = np.nan_to_num(time_series, nan=0.0)
        signal_hybrid60 = np.nan_to_num(signal_hybrid60, nan=0.0)
        peak_envelope = np.nan_to_num(peak_envelope, nan=0.0)

        if self.sequence_length is not None:
            if len(signal_hybrid60) > self.sequence_length:
                # Always select the last self.sequence_length samples
                # time_series = time_series[-self.sequence_length:]
                signal_hybrid60 = signal_hybrid60[-self.sequence_length:]
                peak_envelope = peak_envelope[-self.sequence_length:]
               
            elif len(signal_hybrid60) < self.sequence_length:
                padding_length = self.sequence_length - len(signal_hybrid60)
                # time_series = np.pad(time_series, (padding_length, 0), mode='constant', constant_values=0)
                signal_hybrid60 = np.pad(signal_hybrid60, (padding_length, 0), mode='constant', constant_values=0)
                peak_envelope = np.pad(peak_envelope, (padding_length, 0), mode='constant', constant_values=0)


        outcome_30min = npz_data['labels'][-2]
        if outcome_30min >= 3:
            outcome_30min = 3
        outcome_30min -= 1 


        outcome_bmv = npz_data['labels'][-3]
        if outcome_bmv == 2:
            outcome_bmv = 0 # NOT ATTEMPTED
        else:
            outcome_bmv = 1 # ATTEMPTED

        stacked_signals = np.stack([signal_hybrid60, peak_envelope])
        stacked_signals = torch.from_numpy(stacked_signals).float()

        # time_series = time_series.reshape(self.sequence_length, 1)
        outcome_30min_tensor = torch.tensor(outcome_30min).long()
        outcome_bmv_tensor = torch.tensor(outcome_bmv).long()
        # signal_hybrid60 = torch.from_numpy(signal_hybrid60).float()
        # peak_envelope = torch.from_numpy(peak_envelope).float()

        return stacked_signals, outcome_30min_tensor, outcome_bmv_tensor

class TransformerClassifier_step3_twoInputs(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, patch_size, dropout=0.1):
        super(TransformerClassifier_step3_twoInputs, self).__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding_step3(d_model, max_len=10000)

        if not (0 <= dropout <= 1):
            raise ValueError("Dropout probability must be between 0 and 1")

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.bmv_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, src, mask=None):
        # src is expected to be of shape (batch_size, 2, seq_len)
        batch_size, num_channels, seq_len = src.shape
        num_patches = seq_len // self.patch_size
        if num_patches * self.patch_size != seq_len:
            padding_length = self.patch_size - (seq_len % self.patch_size)
            src = torch.cat((src, torch.zeros(batch_size, num_channels, padding_length, device=src.device)), dim=input_dim)
            seq_len += padding_length

        # Reshape to (batch_size, num_patches, patch_size * num_channels)
        src = src.reshape(batch_size, num_channels, num_patches, self.patch_size)
        src = src.permute(0, 2, 1, 3).contiguous()
        src = src.view(batch_size, num_patches, -1)
        src = src.transpose(0, 1)  # (num_patches, batch_size, patch_size * num_channels)

        src = self.input_proj(src)
        src = self.pos_encoder(src)

        if mask is not None:
            float_mask = mask.float().unsqueeze(1)
            float_mask = float_mask.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
            float_mask = float_mask.transpose(1, 2)
            src = src * (1 - float_mask * 0.8)

        memory = self.transformer_encoder(src)
        last_hidden_state = memory[-1, :, :]
        bmv_pred = self.bmv_classifier(last_hidden_state)
        return bmv_pred


# class FHRDataset_v6_envelope(Dataset):
#     def __init__(self, data_dir, sequence_length=7200):
#         self.data_dir = data_dir
#         self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
#         self.sequence_length = sequence_length

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         file_path = os.path.join(self.data_dir, self.files[idx])
#         npz_data = np.load(file_path)
#         interp_mask = npz_data['interp_mask']
#         peak_envelope = npz_data['peak_env']
#         savgol_envelope = npz_data['savgol_env']
#         signal_hybrid60 = npz_data['inpainted_model_1']

#         # Ensure all arrays are 1-dimensional
#         signal_hybrid60 = signal_hybrid60.flatten()
#         interp_mask = interp_mask.flatten()
#         peak_envelope = peak_envelope.flatten()
#         savgol_envelope = savgol_envelope.flatten()

#         # Ensure all arrays have the same length
#         min_length = min(len(peak_envelope), len(savgol_envelope), len(signal_hybrid60))
#         interp_mask = interp_mask[:min_length]
#         peak_envelope = peak_envelope[:min_length]
#         savgol_envelope = savgol_envelope[:min_length]
#         signal_hybrid60 = signal_hybrid60[:min_length]

#         # Impute NaN values with zeros for all signals
#         interp_mask = np.nan_to_num(interp_mask, nan=0.0)
#         peak_envelope = np.nan_to_num(peak_envelope, nan=0.0)
#         savgol_envelope = np.nan_to_num(savgol_envelope, nan=0.0)
#         signal_hybrid60 = np.nan_to_num(signal_hybrid60, nan=0.0)

#         if self.sequence_length is not None:
#             if len(signal_hybrid60) > self.sequence_length:
#                 interp_mask = interp_mask[-self.sequence_length:]
#                 peak_envelope = peak_envelope[-self.sequence_length:]
#                 savgol_envelope = savgol_envelope[-self.sequence_length:]
#                 signal_hybrid60 = signal_hybrid60[-self.sequence_length:]
#             elif len(signal_hybrid60) < self.sequence_length:
#                 padding_length = self.sequence_length - len(signal_hybrid60)
#                 interp_mask = np.pad(interp_mask, (padding_length, 0), mode='constant', constant_values=0)
#                 peak_envelope = np.pad(peak_envelope, (padding_length, 0), mode='constant', constant_values=0)
#                 savgol_envelope = np.pad(savgol_envelope, (padding_length, 0), mode='constant', constant_values=0)
#                 signal_hybrid60 = np.pad(signal_hybrid60, (padding_length, 0), mode='constant', constant_values=0)

#         # Normalize envelopes if necessary
#         min_val, max_val = 0, 220 
#         if max_val > min_val:
#             peak_envelope = (peak_envelope - min_val) / (max_val - min_val + 1e-8)
#             savgol_envelope = (savgol_envelope - min_val) / (max_val - min_val + 1e-8)

#         # Stack the signals and interpolation mask
#         data = np.stack([signal_hybrid60, peak_envelope, savgol_envelope], axis=-1)
#         # interp_mask = interp_mask.reshape(self.sequence_length, 1)

#         outcome_30min = npz_data['labels'][-2]
#         if outcome_30min >= 3:
#             outcome_30min = 3
#         outcome_30min -= 1 

#         outcome_resus = npz_data['labels'][-6]
#         if outcome_resus == 2:
#             outcome_resus = 0 # NOT ATTEMPTED
#         else:
#             outcome_resus = 1 # ATTEMPTED

#         return torch.from_numpy(data).float(), torch.from_numpy(interp_mask).float(), torch.tensor(outcome_30min).long(), torch.tensor(outcome_resus).long()


class TransformerClassifier_step32(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, patch_size, dropout=0.1):
        super(TransformerClassifier_step32, self).__init__()
        self.patch_size = patch_size
        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding_step3(d_model, max_len=10000)

        # Ensure dropout is a valid value between 0 and 1
        if not (0 <= dropout <= 1):
            raise ValueError("Dropout probability must be between 0 and 1")

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Classifier for resuscitation (binary classification)
        self.resuscitation_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

        # Classifier for outcome (3-class classification)
        self.outcome_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def forward(self, src, mask=None):
        # src is expected to be of shape (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = src.shape
        print(f"Input shape: {src.shape}")

        # Reshape to (batch_size, num_patches, patch_size * input_dim)
        num_patches = seq_len // self.patch_size
        if num_patches * self.patch_size != seq_len:
            # Handle cases where sequence length is not a multiple of patch size
            padding_length = self.patch_size - (seq_len % self.patch_size)
            src = torch.cat((src, torch.zeros(batch_size, padding_length, input_dim, device=src.device)), dim=1)
            seq_len += padding_length
        
        src = src.reshape(batch_size, num_patches, self.patch_size * input_dim)
        print(f"After reshape: {src.shape}")

        src = src.transpose(0, 1)
        print(f"After transpose: {src.shape}")

        # Project input to d_model dimensions
        src = self.input_proj(src)
        print(f"After projection: {src.shape}")
        src = self.pos_encoder(src)

        # Convert and apply boolean/float mask
        if mask is not None:
            float_mask = mask.float().transpose(0, 1).unsqueeze(-1)
            src = src * (1 - float_mask * 0.8)

        memory = self.transformer_encoder(src)
        last_hidden_state = memory[-1, :, :]

        # Predict resuscitation and outcome
        resuscitation_pred = self.resuscitation_classifier(last_hidden_state)
        outcome_pred = self.outcome_classifier(last_hidden_state)

        return resuscitation_pred, outcome_pred

class TransformerClassifier_step3(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, patch_size, dropout=0.1):
        super(TransformerClassifier_step3, self).__init__()
        self.patch_size = patch_size
        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding_step3(d_model, max_len=10000)

        # Ensure dropout is a valid value between 0 and 1
        if not (0 <= dropout <= 1):
            raise ValueError("Dropout probability must be between 0 and 1")

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Classifier for resuscitation (binary classification)
        self.resuscitation_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

        # Classifier for outcome (3-class classification)
        self.outcome_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def forward(self, src, mask=None):
        # src is expected to be of shape (batch_size, seq_len)
        batch_size, seq_len = src.shape
        num_patches = seq_len // self.patch_size
        if num_patches * self.patch_size != seq_len:
            # Handle cases where sequence length is not a multiple of patch size
            padding_length = self.patch_size - (seq_len % self.patch_size)
            src = torch.cat((src, torch.zeros(batch_size, padding_length, device=src.device)), dim=1)
            seq_len += padding_length

        src = src.reshape(batch_size, num_patches, self.patch_size * input_dim).transpose(0, 1)

        # Project input to d_model dimensions
        src = self.input_proj(src)
        src = self.pos_encoder(src)

        if mask is not None:
         # Patch the mask using the same patch size and stride
            float_mask = mask.float().unsqueeze(1)  # Add channel dimension
            float_mask = float_mask.unfold(dimension=1, size=self.patch_size, stride=self.patch_size)  # Patch the mask with stride
            float_mask = float_mask.transpose(1, 2)  # Swap dimensions for element-wise multiplication

            # Apply mask element-wise to src through multiplication
            src = src * (1- float_mask * 0.8)

        memory = self.transformer_encoder(src)

        # Get the last hidden state for classification
        last_hidden_state = memory[-1, :, :]

        # Predict resuscitation and outcome
        resuscitation_pred = self.resuscitation_classifier(last_hidden_state)
        outcome_pred = self.outcome_classifier(last_hidden_state)

        return resuscitation_pred, outcome_pred

class TransformerClassifier_step3_oneClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, patch_size, dropout=0.1):
        super(TransformerClassifier_step3_oneClassifier, self).__init__()
        self.patch_size = patch_size
        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding_step3(d_model, max_len=10000)

        # Ensure dropout is a valid value between 0 and 1
        if not (0 <= dropout <= 1):
            raise ValueError("Dropout probability must be between 0 and 1")

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Classifier for resuscitation (binary classification)
        self.bmv_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, src, mask=None):
        # src is expected to be of shape (batch_size, seq_len)
        batch_size, seq_len = src.shape
        num_patches = seq_len // self.patch_size
        if num_patches * self.patch_size != seq_len:
            # Handle cases where sequence length is not a multiple of patch size
            padding_length = self.patch_size - (seq_len % self.patch_size)
            src = torch.cat((src, torch.zeros(batch_size, padding_length, device=src.device)), dim=1)
            seq_len += padding_length

        src = src.reshape(batch_size, num_patches, self.patch_size * input_dim).transpose(0, 1)

        # Project input to d_model dimensions
        src = self.input_proj(src)
        src = self.pos_encoder(src)

        if mask is not None:
         # Patch the mask using the same patch size and stride
            float_mask = mask.float().unsqueeze(1)  # Add channel dimension
            float_mask = float_mask.unfold(dimension=1, size=self.patch_size, stride=self.patch_size)  # Patch the mask with stride
            float_mask = float_mask.transpose(1, 2)  # Swap dimensions for element-wise multiplication

            # Apply mask element-wise to src through multiplication
            src = src * (1- float_mask * 0.8)

        memory = self.transformer_encoder(src)

        # Get the last hidden state for classification
        last_hidden_state = memory[-1, :, :]

        # Predict resuscitation and outcome
        bmv_pred = self.bmv_classifier(last_hidden_state)
        return bmv_pred

class PositionalEncoding_step3(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding_step3, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RNN_VAE(nn.Module):
    def __init__(self, input_dim=1, sequence_length=7200, hidden_dim=1024, latent_dim=512, num_layers=2):
        super(RNN_VAE, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.encoder_ln = LayerNorm(hidden_dim * 2)

        # Decoder
        self.decoder_rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        self.decoder_ln = LayerNorm(hidden_dim)

    def encode(self, x):
        # Add input shape validation
        if x.dim() != 3 or x.size(1) != self.sequence_length or x.size(2) != self.input_dim:
            raise ValueError(f"Expected input to be of shape (batch_size, {self.sequence_length}, {self.input_dim}), but got {x.shape}")
        
        _, h_n = self.encoder_rnn(x)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_dim)
        h_n = h_n[-1]
        h_n = h_n.transpose(0, 1).contiguous().view(-1, self.hidden_dim * 2)
        h_n = self.encoder_ln(h_n)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        output, _ = self.decoder_rnn(z)
        output = self.decoder_ln(output)
        return self.fc_output(output)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def rnn_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Check for NaN values
    if torch.isnan(recon_loss) or torch.isnan(kld_loss):
        print("Warning: NaN detected in loss calculation")
        return torch.tensor(float('inf'), device=recon_x.device)
    
    return recon_loss + beta * kld_loss




class TransformerMaskedAutoencoder_envelope(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, patch_size, dropout=0.1):
        super(TransformerMaskedAutoencoder_envelope, self).__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.input_proj = nn.Linear(3* patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Change output projection to only predict the time series
        self.output_proj = nn.Linear(d_model, 3 * patch_size)

    def forward(self, src, tgt, mask):
        # src and tgt are expected to be of shape (batch_size, input_dim, seq_len)
        batch_size, input_dim, seq_len = src.shape
        num_patches = seq_len // self.patch_size

        # Reshape to (batch_size, input_dim, num_patches, patch_size)
        src = src.reshape(batch_size, input_dim, num_patches, self.patch_size)
        tgt = tgt.reshape(batch_size, input_dim, num_patches, self.patch_size)
        
        # Permute to (batch_size, num_patches, input_dim, patch_size)
        src = src.permute(0, 2, 1, 3)
        tgt = tgt.permute(0, 2, 1, 3)
        
        # Flatten the last two dimensions
        src = src.reshape(batch_size, num_patches, input_dim * self.patch_size)
        tgt = tgt.reshape(batch_size, num_patches, input_dim * self.patch_size)
        
        # Transpose to (num_patches, batch_size, input_dim * patch_size)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        # Project input to d_model dimensions
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Convert boolean mask to float mask
        float_mask = mask.float().transpose(0, 1).unsqueeze(-1)
        
        # Apply mask to src
        src = src * (1 - float_mask)
        
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.output_proj(output)
        
        # Reshape and permute back to (batch_size, 1, seq_len) for time series only
        output = output.transpose(0, 1)
        output = output.reshape(batch_size, num_patches, 3,  self.patch_size)
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(batch_size, 3, seq_len)
        
        return output

class PositionalEncoding_envelope(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding_envelope, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].expand(-1, x.size(1), -1)

def create_mask_envelope(num_patches, batch_size, mask_ratio=0.15):
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    num_masked = max(1, int(mask_ratio * num_patches))
    for i in range(batch_size):
        start = torch.randint(0, num_patches - num_masked + 1, (1,))
        mask[i, start:start+num_masked] = True
    return mask

def apply_mask_to_data(data, mask, patch_size):
    # data shape: (batch_size, num_channels, seq_len)
    # mask shape: (batch_size, num_patches)
    batch_size, num_channels, seq_len = data.shape
    num_patches = seq_len // patch_size
    
    # Reshape data to (batch_size, num_channels, num_patches, patch_size)
    data_reshaped = data.reshape(batch_size, num_channels, num_patches, patch_size)
    
    # Expand mask to match time series channel shape
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
    
    # Apply mask only to time series channel (assumed to be the first channel)
    data_reshaped[:, 0] = data_reshaped[:, 0].masked_fill(mask_expanded, 0)
    
    # Reshape back to original shape
    masked_data = data_reshaped.reshape(batch_size, num_channels, seq_len)
    
    return masked_data

class TimeSeriesFrequencyLoss_envelope(nn.Module):
    def __init__(self, alpha=1.0):
        super(TimeSeriesFrequencyLoss_envelope, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # Ensure the input is of float type
        pred = pred.float()
        target = target.float()

        # Compute FFT along the time dimension
        pred_fft = fft.fft(pred, dim=2)
        target_fft = fft.fft(target, dim=2)

        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Compute the Focal Frequency Loss
        weight = torch.pow(1 - torch.exp(-torch.abs(target_mag - pred_mag)), self.alpha)
        loss = weight * torch.abs(target_mag - pred_mag)
        
        return torch.mean(loss)






class TimeSeriesFrequencyLoss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(TimeSeriesFrequencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # Ensure the input is of float type
        pred = pred.float()
        target = target.float()

        # Compute FFT
        pred_fft = fft.fft(pred, dim=1)
        target_fft = fft.fft(target, dim=1)

        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Compute the Focal Frequency Loss
        weight = torch.pow(1 - torch.exp(-torch.abs(target_mag - pred_mag)), self.alpha)
        loss = weight * torch.abs(target_mag - pred_mag)
        
        return torch.mean(loss)

# class PositionalEncoding_o(nn.Module):
#     def __init__(self, d_model, max_len=10000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :].expand(-1, x.size(1), -1)

# class TransformerMaskedAutoencoder_o(nn.Module):
#     def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
#         super(TransformerMaskedAutoencoder, self).__init__()
#         self.input_proj = nn.Linear(input_dim, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
#         decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
#         self.output_proj = nn.Linear(d_model, input_dim)

#     def forward(self, src, tgt):
#         # src and tgt are expected to be of shape (batch_size, seq_len, input_dim)
#         # Transpose to (seq_len, batch_size, input_dim)
#         src = src.transpose(0, 1)
#         tgt = tgt.transpose(0, 1)
        
#         # Project input to d_model dimensions
#         src = self.input_proj(src)
#         tgt = self.input_proj(tgt)  # Add this line
        
#         src = self.pos_encoder(src)
#         tgt = self.pos_encoder(tgt)  # Add this line
        
#         memory = self.transformer_encoder(src)
#         output = self.transformer_decoder(tgt, memory)
#         output = self.output_proj(output)
        
#         # Transpose back to (batch_size, seq_len, input_dim)
#         return output.transpose(0, 1)

# def create_mask_o(seq_len, batch_size, mask_ratio=0.15):
#     mask = torch.zeros((batch_size, seq_len, 1), dtype=torch.bool)
#     num_masked = int(mask_ratio * seq_len)
#     for i in range(batch_size):
#         mask_indices = torch.randperm(seq_len)[:num_masked]
#         mask[i, mask_indices, 0] = True
#     return mask

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Calculate the number of convolutions needed
        self.num_conv = 5
        self.conv_output_size = input_dim // (2**self.num_conv)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(512 * self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(512 * self.conv_output_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * self.conv_output_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 512, self.conv_output_size)
        return self.decoder(h)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = x.squeeze(-1)  # Remove the last dimension if it's 1
        x = x.unsqueeze(1)  # Add channel dimension
        mu, logvar = self.encode(x)
        # print("Mu shape:", mu.shape)
        z = self.reparameterize(mu, logvar)
        # print("Z shape:", z.shape)
        decoded = self.decode(z)
        # print("Decoded shape:", decoded.shape)
        return decoded, mu, logvar

def cvae_loss(recon_x, x, mu, logvar, beta=0.9):
    x = x.squeeze(-1)  # Remove the last dimension if it's 1
    x = x.unsqueeze(1)  # Add channel dimension to match recon_x
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


# Loss function
def vae_loss_o(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



def visualize_reconstruction(model, data_loader, device, epoch, save_path):
    with torch.no_grad():
        # Get a random batch
        data, _ = next(iter(data_loader))
   
        # sample_idx = random.randint(0, data.shape[0] - 1)
        # sample = data[sample_idx].unsqueeze(0).to(device)
        sample = data[5].unsqueeze(0).to(device) # 5 is random sample can be changed
        
        # Get the reconstruction
        recon, _, _ = model(sample)
        
        # Move tensors to CPU and convert to numpy arrays
        original_nomr = sample.cpu().squeeze().numpy()
        reconstructed = recon.cpu().squeeze().numpy()
        
        # Plot
        plt.figure(figsize=(24, 8))
        plt.plot(original_nomr, label='Original-Norm', alpha=0.7)
        plt.plot(reconstructed, label='Reconstructed', alpha=0.8)
        plt.title(f'Original vs Reconstructed Signal - Epoch {epoch}')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        
        # Save the plot
        plt.savefig(f'{save_path}/reconstruction_epoch_{epoch+1}.png')
        plt.close()

def visualize_reconstruction_transformer(model, data_loader, device, epoch, save_path, sample_index=5):
    model.eval()
    with torch.no_grad():
        # Get a batch and select a single sample
        data, _ = next(iter(data_loader))
        sample = data[sample_index].unsqueeze(0).to(device)
        # sample = data[sample_index].to(device)
        batch_size, seq_len, input_dim = sample.shape

        # Create mask for this sample
        num_patches = seq_len // model.patch_size
        mask = create_mask(num_patches, 1).to(device)

        # Forward pass
        output = model(sample, sample, mask)

        # Expand mask to match data dimensions
        expanded_mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size).view(1, seq_len, input_dim)

        # Move tensors to CPU and convert to numpy arrays
        original = sample.cpu().squeeze().numpy()
        reconstructed = output.cpu().squeeze().numpy()
        mask_np = expanded_mask.cpu().squeeze().numpy()

        # Create the plot
        fig, ax = plt.subplots(figsize=(24, 8))
        
        # Plot original signal
        ax.plot(original, label='Original', color='blue', alpha=0.7)
        
        # Plot reconstructed signal
        ax.plot(reconstructed, label='Reconstructed', color='red', alpha=0.9)
        
        # Highlight masked regions
        masked_regions = np.where(mask_np == 1)[0]
        for start, end in get_continuous_regions(masked_regions):
            ax.axvspan(start, end, color='gray', alpha=0.4)
        
        # Add labels and title
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Amplitude', fontsize=18)
        ax.set_title(f'Original vs Reconstructed Signal - Epoch {epoch+1}', fontsize=18)
        ax.legend(fontsize=10)
        
        # Add MSE to the plot
        mse = np.mean((original - reconstructed)**2)
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add text explaining the shaded areas
        ax.text(0.02, 0.02, 'Gray areas: Masked regions the model had to reconstruct', 
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(f'{save_path}/reconstruction_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_reconstruction_transformer_envelope(model, data_loader, device, epoch, save_path, sample_index=5):
    model.eval()
    with torch.no_grad():
        # Get a batch and select a single sample
        data, _ = next(iter(data_loader))
        sample = data[sample_index].unsqueeze(0).to(device)
        batch_size, num_channels, seq_len = sample.shape

        # Create mask for this sample
        num_patches = seq_len // model.patch_size
        mask = create_mask(num_patches, 1).to(device)

        # Forward pass
        output = model(sample, sample, mask)

        # Expand mask to match data dimensions
        expanded_mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size).view(1, num_channels, seq_len)

        # Move tensors to CPU and convert to numpy arrays
        original = sample.cpu().squeeze().numpy()
        reconstructed = output.cpu().squeeze().numpy()
        mask_np = expanded_mask.cpu().squeeze().numpy()

        # Create the plot
        fig, axs = plt.subplots(3, 1, figsize=(24, 18), sharex=True)
        fig.suptitle(f'Original vs Reconstructed Signal - Epoch {epoch+1}', fontsize=22)
        
        channel_names = ['FHR', 'Peak Envelope', 'Savgol Envelope']
        colors = ['blue', 'green', 'red']

        for i, (ax, channel_name) in enumerate(zip(axs, channel_names)):
            # Plot original signal
            ax.plot(original[i], label='Original', color=colors[i], alpha=0.7)
            
            # Plot reconstructed signal
            ax.plot(reconstructed[i], label='Reconstructed', color='orange', alpha=0.9)
            
            # Highlight masked regions
            masked_regions = np.where(mask_np[i] == 1)[0]
            for start, end in get_continuous_regions(masked_regions):
                ax.axvspan(start, end, color='gray', alpha=0.4)
            
            # Add labels
            ax.set_ylabel(f'{channel_name}\nAmplitude', fontsize=14)
            ax.legend(fontsize=10)
            
            # Add MSE to the plot
            mse = np.mean((original[i] - reconstructed[i])**2)
            ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                    verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Add x-label to the bottom subplot
        axs[-1].set_xlabel('Time', fontsize=14)
        
        # Add text explaining the shaded areas
        fig.text(0.02, 0.02, 'Gray areas: Masked regions the model had to reconstruct', 
                 fontsize=10, verticalalignment='bottom', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(f'{save_path}/reconstruction_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

def get_continuous_regions(indices):
    regions = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            regions.append((start, indices[i-1] + 1))
            start = indices[i]
    regions.append((start, indices[-1] + 1))
    return regions

def visualize_reconstruction_transformer_envelope(model, data_loader, device, epoch, save_path, sample_index=5):
    model.eval()
    with torch.no_grad():
        # Get a batch and select a single sample
        data, _ = next(iter(data_loader))

        sample = data[sample_index][0].unsqueeze(0).to(device)
        batch_size, seq_len, input_dim = sample.shape

        # Create mask for this sample
        num_patches = seq_len // model.patch_size
        mask = create_mask(num_patches, 1).to(device)

        # Forward pass
        output = model(sample, sample, mask)

        # Expand mask to match data dimensions
        expanded_mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size).view(1, seq_len, input_dim)

        # Move tensors to CPU and convert to numpy arrays
        original = sample.cpu().squeeze().numpy()
        reconstructed = output.cpu().squeeze().numpy()
        mask_np = expanded_mask.cpu().squeeze().numpy()

        # Create the plot
        fig, ax = plt.subplots(figsize=(24, 8))
        
        # Plot original signal
        ax.plot(original, label='Original', color='blue', alpha=0.7)
        
        # Plot reconstructed signal
        ax.plot(reconstructed, label='Reconstructed', color='red', alpha=0.9)
        
        # Highlight masked regions
        masked_regions = np.where(mask_np == 1)[0]
        for start, end in get_continuous_regions(masked_regions):
            ax.axvspan(start, end, color='gray', alpha=0.4)
        
        # Add labels and title
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Amplitude', fontsize=18)
        ax.set_title(f'Original vs Reconstructed Signal - Epoch {epoch+1}', fontsize=18)
        ax.legend(fontsize=10)
        
        # Add MSE to the plot
        mse = np.mean((original - reconstructed)**2)
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add text explaining the shaded areas
        ax.text(0.02, 0.02, 'Gray areas: Masked regions the model had to reconstruct', 
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        plt.savefig(f'{save_path}/reconstruction_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
def get_continuous_regions(arr):
    # Helper function to get start and end indices of continuous regions
    regions = []
    start = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1] + 1:
            regions.append((start, arr[i-1] + 1))
            start = arr[i]
    regions.append((start, arr[-1] + 1))
    return regions

def plot_reconstructions(data, reconstructed, fnames, norm, save_path, n=3):
    original = data[:n]
    reconstructed = reconstructed[:n]
    file_names = fnames[:n]

    fig, axes = plt.subplots(n, 1, figsize=(15, 4*n))
    for i in range(n):
        ax = axes[i] if n > 1 else axes
        
        # Reshape if necessary
        orig = original[i].reshape(-1)
        recon = reconstructed[i].reshape(-1)
        
        ax.plot(orig, label='Original', alpha=0.8)
        ax.plot(recon, label='Reconstructed', alpha=0.7)

        ax.set_title(file_names[i], fontsize=24)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

        mse = np.mean((orig - recon)**2)
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/final_reconstructions_with_{norm}.png', dpi=50, bbox_inches='tight')
    plt.close()

def plot_masks(masks, save_path, patch_size=60):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, mask in enumerate(masks[:5]):  # Plot first 9 masks
        mask = mask.squeeze()
        
        # Reshape the mask to show patches
        num_patches = len(mask) // patch_size
        reshaped_mask = mask[:num_patches * patch_size].reshape(num_patches, patch_size)
        
        # Plot the mask
        im = axes[i].imshow(reshaped_mask, cmap='binary', aspect='auto', interpolation='nearest')
        axes[i].set_title(f'Mask {i+1}')
        axes[i].set_xlabel('Time steps within patch')
        axes[i].set_ylabel('Patch number')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_reconstructions_v1(data, reconstructed, fnames, norm, save_path, n=3,  force_norm=True):

    original = data[:n+1]
    reconstructed = reconstructed[:n+1]
    file_names = fnames[:n+1]

    fig, axes = plt.subplots(n, 1, figsize=(15, 4*n))
    for i in range(n):
        ax = axes[i] if n > 1 else axes

        ax.plot(original[i], label='Original', alpha=0.8)
        
        # Plot reconstruction
        ax.plot(reconstructed[i], label='Reconstructed', alpha=0.7)

        ax.set_title(file_names[i], fontsize=24)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

        mse = np.mean((original[i] - reconstructed[i])**2)
        ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/final_reconstructions_with_{norm}.png', dpi=50, bbox_inches='tight')
    plt.close()


def plot_reconstructions_v2(data, reconstructed, fnames, norm, save_path, force_norm=True, n=5):
    # Determine the number of samples to plot
    n = min(n, len(data), len(reconstructed), len(fnames))
    
    if n == 0:
        print("No data available for plotting reconstructions.")
        return

    original = data[:n]
    reconstructed = reconstructed[:n]
    file_names = fnames[:n]

    fig, axes = plt.subplots(n, 1, figsize=(20, 6*n))
    axes = [axes] if n == 1 else axes  # Ensure axes is always a list

    for i in range(n):
        ax = axes[i]

        signal = original[i]

        if force_norm: # Fix this for entire dataset normalized in same way
            min_val= 0
            max_val=200
            if max_val > min_val:
                signal = (signal - min_val) / (max_val - min_val)
            else:
                signal = np.zeros_like(signal)
        
        # Plot original
        ax.plot(signal, label='Original', alpha=0.7, color='blue')
        
        # Plot reconstruction
        ax.plot(reconstructed[i], label='Reconstructed', alpha=0.7, color='red')

        ax.set_title(file_names[i], fontsize=18)
        
        # Create legend for each subplot
        ax.legend()
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/final_reconstructions_with_{norm}.png')
    plt.close()

def collate_fn(batch):
    data, mask,  labels_30min, labels_resus = zip(*batch)
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    mask = rnn_utils.pad_sequence(mask, batch_first=True, padding_value=0)
    # mask = torch.stack(mask)
    labels_30min = torch.stack(labels_30min)
    labels_resus = torch.stack(labels_resus)
    return data, mask, labels_30min, labels_resus

def collate_fn_twoInputs(batch):
    data, labels_30min, labels_resus = zip(*batch)
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    labels_30min = torch.stack(labels_30min)
    labels_resus = torch.stack(labels_resus)
    return data, labels_30min, labels_resus

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target):
        # Move the weight tensor to the same device as the input tensor
        if self.weight is not None:
            self.weight = self.weight.to(input.device)

        # Compute the cross-entropy loss
        log_pt = F.log_softmax(input, dim=-1)
        pt = torch.exp(log_pt)
        ce_loss = F.nll_loss(log_pt, target, weight=self.weight, reduction='none')

        # Compute the focal loss
        pt = pt.gather(1, target.unsqueeze(1))
        alpha = torch.pow((1 - pt), self.gamma)
        focal_loss = alpha * ce_loss

        return focal_loss
        
class TCNModel(nn.Module):
    def __init__(self, input_size, num_classes_30min, num_classes_24hours, num_resus_classes, num_channels, kernel_size, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**0, dilation=2**0),
            nn.BatchNorm1d(num_channels),  # Add BatchNorm1d layer
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**1, dilation=2**1),
            nn.BatchNorm1d(num_channels),  # Add BatchNorm1d layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**2, dilation=2**2),
            nn.BatchNorm1d(num_channels),  # Add BatchNorm1d layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**3, dilation=2**3),
            nn.BatchNorm1d(num_channels),  # Add BatchNorm1d layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**4, dilation=2**4),
            nn.BatchNorm1d(num_channels),  # Add BatchNorm1d layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveMaxPool1d(1) # (batch_size, num_channels, 1).
        )
        self.classifier_30min = nn.Linear(num_channels, num_classes_30min)
        self.classifier_24hours = nn.Linear(num_channels, num_classes_24hours)
        self.classifier_resus = nn.Linear(num_channels, num_resus_classes)

    def forward(self, x):
        # Assuming x has shape (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_size, sequence_length)

        x = self.tcn(x)
        x = x.squeeze(-1)  # Remove the channel dimension , make (batch_size, num_channels)
        # print("Shape of feature," , x.shape)
        outcome_30min = self.classifier_30min(x)
        outcome_24hours = self.classifier_24hours(x)
        outcome_resus = self.classifier_resus(x)

        # Apply log-softmax and subtract 1 from target labels
        # log_probs_30min = F.log_softmax(outcome_30min, dim=1)
        # log_probs_24hours = F.log_softmax(outcome_24hours, dim=1)

        return outcome_30min, outcome_24hours, outcome_resus

class TCNModelv2(nn.Module):
    def __init__(self, input_size, num_classes_30min, num_classes_24hours, num_resus_classes, num_channels, kernel_size, dropout=0.2):
        super(TCNModelv2, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**0, dilation=2**0),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**1, dilation=2**1),
            nn.BatchNorm1d(num_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels * 2, num_channels * 4, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**2, dilation=2**2),
            nn.BatchNorm1d(num_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels * 4, num_channels * 4, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**3, dilation=2**3),
            nn.BatchNorm1d(num_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels * 4, num_channels * 4, kernel_size=kernel_size, padding=(kernel_size - 1) * 2**4, dilation=2**4),
            nn.BatchNorm1d(num_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveMaxPool1d(1)
        )
        self.attention = nn.MultiheadAttention(num_channels * 4, num_heads=4, batch_first=True)
        self.classifier_30min = nn.Sequential(
            nn.Linear(num_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes_30min)
        )
        self.classifier_24hours = nn.Sequential(
            nn.Linear(num_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes_24hours)
        )
        self.classifier_resus = nn.Sequential(
            nn.Linear(num_channels * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_resus_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # Assuming x has shape (batch_size, sequence_length, input_size)
        # x = x.permute(0, 2, 1)  # Permute to (batch_size, input_size, sequence_length)

        x = self.tcn(x)
        x = x.transpose(1, 2)  # Permute to (batch_size, sequence_length, num_channels)
        x, _ = self.attention(x, x, x)  # Appl self-attention
        x = x.transpose(1, 2)  # Permute back to (batch_size, num_channels, sequence_length)
        x = x.squeeze(-1)  # Remove the channel dimension

        outcome_30min = self.classifier_30min(x)
        outcome_24hours = self.classifier_24hours(x)
        outcome_resus = self.classifier_resus(x)

        return outcome_30min, outcome_24hours, outcome_resus

def make_pretty_cm(cf, group_names=None, categories='auto', count=True,
                   percent=True, cbar=True, xyticks=True, xyplotlabels=True, sum_stats=True,
                   figsize=None, cmap='Blues', title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]
    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))
        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.set(font_scale=1.5)
    # sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=group_names if group_names else categories,
            yticklabels=group_names if group_names else categories)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.savefig(f"{title}.png")


def make_cm(y_true, y_pred, classes):
    # labels = [‘True Neg’,’False Pos’,’False Neg’,’True Pos’]
    cm = confusion_matrix(y_true, y_pred)
    confusion_matrix_df = pd.DataFrame(cm, columns=classes)
    fig = plt.figure(figsize=(14, 14))
    fig = sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="coolwarm")
    fig.set(ylabel="True", xlabel="Predicted", title='DKL predictions')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    return cm

def plot_three_signals(signal1, signal3, signal1_label, signal3_label, title, x_label, y_label):
    """
    Plots three signals on the same plot with a legend.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(signal1, label=signal1_label)
    ax.plot(signal3, label=signal3_label)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(fontsize=14)
    plt.show()

    def detect_noise(self, verbose=1, max_change_allowed=30, max_segment=50, max_halfdouble_threshold=5):
        # Function to detect and remove noise in the FHR signal
        signal = self.fhr

        # Initialization of required variables
        removed_values = np.zeros(np.shape(signal))
        doublehalfing = np.zeros(np.shape(signal))
        removed_areas = 0
        removed_mhr = np.zeros(np.shape(signal))

        # Fill gaps using forward zero-hold
        fhr_to_filter = signal
        removed_values, doublehalfing, removed_areas, removed_mhr = _remove_gaps(
            fhr_to_filter=fhr_to_filter,
            mhr=self.mhr,
            method='forward',
            max_change_allowed=max_change_allowed,
            max_segment=max_segment,
            max_halvdouble_threshold=max_halfdouble_threshold,
            removed_values=removed_values,
            removed_areas=removed_areas,
            removed_mhr=removed_mhr,
            doublehalving=doublehalfing
        )

        # Fill gaps using backward zero-hold
        fhr_to_filter = signal
        removed_values, doublehalfing, removed_areas, removed_mhr = _remove_gaps(
            fhr_to_filter=fhr_to_filter,
            mhr=self.mhr,
            method='forward',
            max_change_allowed=max_change_allowed,
            max_segment=max_segment,
            max_halvdouble_threshold=max_halfdouble_threshold,
            removed_values=removed_values,
            removed_areas=removed_areas,
            removed_mhr=removed_mhr,
            doublehalving=doublehalfing,
        )

        # Create cleaned and interpolated version
        clean_fhr = signal
        clean_fhr[removed_values == 1] = 0
        clean_fhr[doublehalfing > 0] = doublehalfing[doublehalfing > 0] * clean_fhr[doublehalfing > 0]
        trustworthy_signal = np.zeros(np.shape(clean_fhr))
        trustworthy_signal[np.nonzero(clean_fhr)] = 1
        clean_fhr = np.interp(self.timecode_fhr,
                              self.timecode_fhr[trustworthy_signal == 1],
                              clean_fhr[trustworthy_signal == 1])

        # Set cleaned information in object
        self.cleaninfo_cleanFHR = clean_fhr
        self.cleaninfo_trustworthyFHR = trustworthy_signal
        self.cleaninfo_missingSamples = np.concatenate(signal == 0).astype(int)
        self.cleaninfo_noise = np.concatenate(removed_values.astype(int))
        self.cleaninfo_interpolated = np.bitwise_or(self.cleaninfo_missingSamples > 0,
                                                    np.concatenate(removed_values) > 0)
        self.cleaninfo_doubleHalfing = doublehalfing
        self.cleaninfo_likelyMHR = removed_mhr.astype(int)
        self.cleaninfo_usedParams_maxChangeAllowed = max_change_allowed
        self.cleaninfo_usedParams_maxSegment = max_segment
        self.cleaninfo_version = '1.3.2'

def _replace_zeros(fhr_to_filter, method='forward'):
    if method == 'forward':
        for i in range(1, len(fhr_to_filter) - 1, 1):
            if fhr_to_filter[i] == 0:
                fhr_to_filter[i] = fhr_to_filter[i - 1]
    else:
        for i in range(len(fhr_to_filter) - 2, 0, -1):
            if fhr_to_filter[i] == 0:
                fhr_to_filter[i] = fhr_to_filter[i + 1]
    return fhr_to_filter

def _remove_gaps(fhr_to_filter,
                 mhr,
                 method,
                 max_change_allowed,
                 max_segment,
                 max_halvdouble_threshold,
                 removed_values,
                 removed_areas,
                 removed_mhr,
                 doublehalving):
    if np.count_nonzero(doublehalving):
        fhr_to_filter[np.nonzero(doublehalving)] = fhr_to_filter[np.nonzero(doublehalving)] * \
                                                   doublehalving[np.nonzero(doublehalving)]
    if np.count_nonzero(removed_values):
        fhr_to_filter[np.nonzero(removed_values)] = 0
    if np.count_nonzero(removed_mhr):
        fhr_to_filter[np.nonzero(removed_mhr)] = 0

    # Fill zeros in all gaps
    fhr_to_filter = _replace_zeros(fhr_to_filter=fhr_to_filter, method=method)

    while True:
        # Identify drops and increases
        signal_diff = np.diff(fhr_to_filter.astype(np.int16), axis=0)
        signal_drops = np.array([0, 0], ndmin=2)
        signal_increases = np.array([0, 0], ndmin=2)

        for i in range(len(signal_diff)):
            if signal_diff[i] < -max_change_allowed:
                signal_drops = np.vstack((signal_drops, np.array([i, 0])))
                if not signal_increases[signal_increases.shape[0] - 1, 1]:
                    signal_increases[signal_increases.shape[0] - 1, 1] = i
            if signal_diff[i] > max_change_allowed:
                if not signal_drops[signal_drops.shape[0] - 1, 1]:
                    signal_drops[signal_drops.shape[0] - 1, 1] = i
                signal_increases = np.vstack((signal_increases, np.array([i, 0])))

        # Remove rows with zeros
        signal_drops = signal_drops[np.all(signal_drops != 0, axis=1)]
        signal_increases = signal_increases[np.all(signal_increases != 0, axis=1)]
        signal_changes = np.vstack((signal_drops, signal_increases))
        signal_changes_length = np.diff(signal_changes, axis=1)

        # Sort rows by size
        sort_idx = signal_changes_length.argsort(axis=0)
        signal_changes[:] = [signal_changes[i] for i in sort_idx]
        signal_changes_length[:] = [signal_changes_length[i] for i in sort_idx]

        # If no changes in FHR are found, exit
        if not len(signal_changes_length):
            break

        removed_areas = removed_areas + 1
        # Only remove segments outside of the normal region
        while True:
            if len(signal_changes_length) > 0:
                if (np.median(fhr_to_filter[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1]) < 160) & (
                        np.median(fhr_to_filter[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1]) > 120):
                    signal_changes = np.delete(signal_changes, 0, axis=0)
                    signal_changes_length = np.delete(signal_changes_length, 0, axis=0)
                else:
                    break
            else:
                break

        # If shortest segment is longer than max_length, then finish. Else remove segment and start over
        if len(signal_changes_length) > 0:
            if signal_changes_length[0] > max_segment:
                break
            else:
                window_length = 60
                start_idx = int(np.maximum(1, signal_changes[0, 0] + 1 - np.fix(window_length / 2)))
                end_idx = int(np.minimum(len(fhr_to_filter), signal_changes[0, 1] + np.fix(window_length / 2)))
                mhr_values = mhr[start_idx:end_idx]

                # Check if FHR is actually MHR
                if np.any(np.abs(np.median(mhr_values[np.any(mhr_values)])) - np.median(
                        fhr_to_filter[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1]) < 5):
                    fhr_to_filter[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 0
                    removed_values[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 1
                    doublehalving[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 0
                    removed_mhr[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 1
                # Check if signal is doubled
                elif (np.abs(fhr_to_filter[signal_changes[0, 0] + 1] / 2 - fhr_to_filter[
                    signal_changes[0, 0]]) < max_halvdouble_threshold) & (np.abs(
                    fhr_to_filter[signal_changes[0, 1]] / 2 - fhr_to_filter[
                        signal_changes[0, 1] + 1]) < max_halvdouble_threshold) & (
                        fhr_to_filter[signal_changes[0, 0]] < fhr_to_filter[signal_changes[0, 0] + 1]):
                    doublehalving[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 0.5
                    fhr_to_filter[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = fhr_to_filter[
                                                                                       signal_changes[0, 0] + 1:
                                                                                       signal_changes[0, 1] + 1] * 0.5

                # Check if signal is halved
                elif np.abs(fhr_to_filter[signal_changes[0, 0] + 1] * 2 - fhr_to_filter[
                    signal_changes[0, 0] + 1]) < max_halvdouble_threshold & np.abs(
                    fhr_to_filter[signal_changes[0, 1] + 1] * 2 - fhr_to_filter[
                        signal_changes[0, 1] + 1]) < max_halvdouble_threshold & fhr_to_filter[
                    signal_changes[0, 0] + 1] > fhr_to_filter[signal_changes[0, 0] + 1]:
                    doublehalving[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 2
                    fhr_to_filter[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = fhr_to_filter[
                                                                                       signal_changes[0, 0] + 1:
                                                                                       signal_changes[0, 1] + 1] * 2

                # Otherwise, remove segment
                else:
                    # tmp = (fhr_to_filter[signal_changes[0,0]+1:signal_changes[0,1]+1])
                    fhr_to_filter[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 0
                    removed_values[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 1
                    doublehalving[signal_changes[0, 0] + 1:signal_changes[0, 1] + 1] = 0

        # Fill in new zeros
        fhr_to_filter = _replace_zeros(fhr_to_filter, method)

    return removed_values, doublehalving, removed_areas, removed_mhr

# # # Create a list of MAT file names
# mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
# print("Total files in directory: ", len(mat_files))
#
# # Load the Excel sheet
# df = pd.read_excel(excel_path)
#
# matched_data = pd.DataFrame(columns=['filename', 'Apgar_5min', 'Resuscitation', 'Stimulation', 'Suction', 'BMV', 'Outcome_30min', 'Outcome_24hours'])
#
# for mat_file in mat_files:
#     # Check if the file name exists in the "NewName" or "episodeName" column
#     mask = df['newName'].str.contains(mat_file.split('.')[0], case=False) | df['episodeName'].str.contains(mat_file.split('.')[0], case=False)
#
#     # If the file name exists, append the relevant columns to the matched_data DataFrame
#     if mask.any():
#         file_data = df.loc[mask, ['Apgar_5min', 'Resuscitation', 'Stimulation', 'Suction', 'BMV', 'Outcome_30min', 'Outcome_24hours']]
#         file_data.insert(0, 'filename', mat_file.split('.')[0])
#         matched_data = pd.concat([matched_data, file_data], ignore_index=True)
#
# # Save the matched data to a new Excel file
# matched_data.to_excel(os.path.join(sav_dir, 'matched_details.xlsx'), index=False)


def convert_batch_list(lst_of_lst):
    return sum(lst_of_lst, [])

def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, epsilon=1e-8):
    # Ensure y_pred and y_true are of the same shape
    y_pred = torch.softmax(y_pred, dim=1)
    y_true = y_true.float()

    # Calculate true positives, false positives, and false negatives
    TP = torch.sum(y_true * y_pred, dim=0)
    FP = torch.sum(y_pred * (1 - y_true), dim=0)
    FN = torch.sum((1 - y_pred) * y_true, dim=0)

    # Calculate Tversky index
    numerator = TP + epsilon
    denominator = TP + alpha * FP + beta * FN + epsilon
    tversky_index = numerator / denominator

    # Calculate Tversky loss
    tversky_loss = 1 - tversky_index

    # Return mean Tversky loss over all classes
    return tversky_loss.mean()

def tversky_focal_loss(y_pred, y_true, alpha=0.7, beta=0.3, gamma=2, epsilon=1e-8):
    # Ensure y_pred and y_true are of the same shape
    y_pred = torch.softmax(y_pred, dim=1)
    y_true = y_true.float()

    # One-hot encoding for multi-class problems
    if len(y_true.shape) == 1:
        y_true = F.one_hot(y_true, num_classes=y_pred.shape[1])

    # Calculate true positives, false positives, and false negatives for each class
    TP = torch.sum(y_true * y_pred, dim=0)
    FP = torch.sum(y_pred * (1 - y_true), dim=0)
    FN = torch.sum((1 - y_pred) * y_true, dim=0)

    # Calculate Tversky index for each class
    numerator = TP + epsilon
    denominator = TP + alpha * FP + beta * FN + epsilon
    tversky_index = numerator / denominator

    # Calculate Focal Tversky loss
    focal_tversky_loss = (1 - tversky_index) ** gamma

    # Return mean Focal Tversky loss over all classes
    return focal_tversky_loss.mean()

def calculate_fid_v2(real_features, generated_features):
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    return fid

def calculate_psnr_v2(real, generated):
    mse = np.mean((real - generated) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
                
def interpolate_missing_values_older(df, model, device, window_size=3600, step_size=30):
    model = model.to(device)
    model.eval()
    
    inpainted_signal = df['fhr_original'].copy().values
    missing_mask = df['missing_mask'].values
    
    # Find all ranges of missing values
    missing_ranges = []
    start_index = None
    for i, is_missing in enumerate(missing_mask):
        if is_missing and start_index is None:
            start_index = i
        elif not is_missing and start_index is not None:
            missing_ranges.append((start_index, i))
            start_index = None
    if start_index is not None:
        missing_ranges.append((start_index, len(df)))
    
    # Inpaint each range recursively
    for start_index, end_index in missing_ranges:
        # Prepare input for the model
        input_data = inpainted_signal[max(0, start_index - window_size):start_index]
        input_data = input_data[~np.isnan(input_data)]  # Remove any NaN values
        
        if len(input_data) == 0:
            continue  # Skip if no valid input data
        
        input_data = np.pad(input_data, (max(0, window_size - len(input_data)), 0), mode='edge')
        
        # Ensure input_data is divisible by patch_size
        if len(input_data) % model.patch_size != 0:
            pad_length = model.patch_size - (len(input_data) % model.patch_size)
            input_data = np.pad(input_data, (pad_length, 0), 'edge')
        
        # Forecast missing values
        remaining_length = end_index - start_index
        forecasted_values = []
        
        while remaining_length > 0:
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                output = model(input_tensor, input_tensor)
                forecast = output[0, -step_size:, 0].cpu().numpy()
            
            forecasted_values.extend(forecast[:min(step_size, remaining_length)])
            remaining_length -= step_size
            
            # Update input_data for next iteration
            input_data = np.concatenate([input_data[step_size:], forecast[:step_size]])
        
        # Fill in the missing values
        inpainted_signal[start_index:end_index] = forecasted_values[:end_index - start_index]
    
    return inpainted_signal

def interpolate_missing_values(df, model, device, window_size=3600, step_size=30):
    model = model.to(device)
    model.eval()
    
    inpainted_signal = df['fhr_original'].copy().values
    missing_mask = df['missing_mask'].values
    signal_length = len(inpainted_signal)
    
    # Find all ranges of missing values
    missing_ranges = []
    start_index = None
    for i, is_missing in enumerate(missing_mask):
        if is_missing and start_index is None:
            start_index = i
        elif not is_missing and start_index is not None:
            missing_ranges.append((start_index, i))
            start_index = None
    if start_index is not None:
        missing_ranges.append((start_index, signal_length))
    
    # Inpaint each range recursively
    for start_index, end_index in missing_ranges:
        # Ensure we don't go beyond the signal length
        end_index = min(end_index, signal_length)
        
        # Prepare input for the model
        input_data = inpainted_signal[max(0, start_index - window_size):start_index]
        input_data = input_data[~np.isnan(input_data)]  # Remove any NaN values
        
        if len(input_data) == 0:
            continue  # Skip if no valid input data
        
        input_data = np.pad(input_data, (max(0, window_size - len(input_data)), 0), mode='edge')
        
        # Ensure input_data is divisible by patch_size
        if len(input_data) % model.patch_size != 0:
            pad_length = model.patch_size - (len(input_data) % model.patch_size)
            input_data = np.pad(input_data, (pad_length, 0), 'edge')
        
        # Forecast missing values
        remaining_length = end_index - start_index
        forecasted_values = []
        
        while remaining_length > 0:
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                output = model(input_tensor, input_tensor)
                forecast = output[0, -step_size:, 0].cpu().numpy()
            
            forecasted_values.extend(forecast[:min(step_size, remaining_length)])
            remaining_length -= step_size
            
            # Update input_data for next iteration
            input_data = np.concatenate([input_data[step_size:], forecast[:step_size]])
            
            # Stop if we've reached the end of the signal
            if start_index + len(forecasted_values) >= signal_length:
                break
        
        # Fill in the missing values
        end_fill = min(end_index, start_index + len(forecasted_values))
        inpainted_signal[start_index:end_fill] = forecasted_values[:end_fill - start_index]
    
    return inpainted_signal[:signal_length]

def interpolate_missing_values_v2(df, model, device, window_size=3600, step_size=30):
    model = model.to(device)
    model.eval()
    
    interpolated_signal = df['time_series'].copy().values
    missing_mask = df['interp_mask'].values
    signal_length = len(interpolated_signal)
    
    # Find the actual length of the signal excluding trailing zeros
    non_zero_indices = np.where(interpolated_signal != 0)[0]
    if len(non_zero_indices) == 0:
        return interpolated_signal  # No non-zero values, return as is
    actual_signal_length = non_zero_indices[-1] + 1
    
    # Find all ranges of missing values within the actual signal length
    missing_ranges = []
    start_index = None
    for i, is_missing in enumerate(missing_mask[:actual_signal_length]):
        if is_missing and start_index is None:
            start_index = i
        elif not is_missing and start_index is not None:
            missing_ranges.append((start_index, i))
            start_index = None
    if start_index is not None:
        missing_ranges.append((start_index, actual_signal_length))
    
    # Inpaint each range recursively
    for start_index, end_index in missing_ranges:
        # Prepare input for the model
        input_data = interpolated_signal[max(0, start_index - window_size):start_index]
        input_data = input_data[~np.isnan(input_data)]  # Remove any NaN values
        
        if len(input_data) == 0:
            continue  # Skip if no valid input data
        
        input_data = np.pad(input_data, (max(0, window_size - len(input_data)), 0), mode='edge')
        
        # Ensure input_data is divisible by patch_size
        if len(input_data) % model.patch_size != 0:
            pad_length = model.patch_size - (len(input_data) % model.patch_size)
            input_data = np.pad(input_data, (pad_length, 0), 'edge')
        
        # Forecast missing values
        remaining_length = end_index - start_index
        forecasted_values = []
        
        while remaining_length > 0:
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                output = model(input_tensor, input_tensor)
                forecast = output[0, -step_size:, 0].cpu().numpy()
            
            forecasted_values.extend(forecast[:min(step_size, remaining_length)])
            remaining_length -= step_size
            
            # Update input_data for next iteration
            input_data = np.concatenate([input_data[step_size:], forecast[:step_size]])
            
            # Stop if we've reached the end of the actual signal length
            if start_index + len(forecasted_values) >= actual_signal_length:
                break
        
        # Fill in the missing values
        end_fill = min(end_index, start_index + len(forecasted_values))
        interpolated_signal[start_index:end_fill] = forecasted_values[:end_fill - start_index]

    # Ensure the output signal has the same length as the original
    interpolated_signal = np.pad(interpolated_signal[:actual_signal_length], (0, signal_length - actual_signal_length), 'constant')
    
    return interpolated_signal



def interpolate_missing_values_v3(df, model, device):
    model = model.to(device)
    model.eval()
    
    time_series = df['time_series'].values
    missing_mask = df['interp_mask'].values.astype(bool)
    signal_length = len(time_series)
    # print("Length of Interpolated signal: ", signal_length)
    
    # Prepare input data for reconstruction
    input_data = time_series.copy()

    # Ensure the length of input_data is divisible by patch_size
    patch_size = model.patch_size
    if signal_length % patch_size != 0:
        pad_length = patch_size - (signal_length % patch_size)
        input_data = np.pad(input_data, (0, pad_length), 'constant')
        missing_mask = np.pad(missing_mask, (0, pad_length), 'constant')

    # Check for NaNs in input data
    assert not np.isnan(input_data).any(), "Input data contains NaNs"

    # Convert to tensor
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(-1).to(device)

    # Debugging: Print shapes
    # print(f"Input tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        output = model(input_tensor, input_tensor)
        reconstructed_signal = output.squeeze().cpu().numpy()
        assert not torch.isnan(output).any(), "Output contains NaNs"

    # print("Length of reconstructed signal: ", len(reconstructed_signal))
    
    # Trim the output back to the original signal length
    reconstructed_signal = reconstructed_signal[:signal_length]
    
    # Replace only the originally missing values with predictions
    inpainted_signal = time_series.copy()
    inpainted_signal[missing_mask] = reconstructed_signal[missing_mask]

    # Calculate MSE for interpolated values at missing mask locations
    mse = np.mean((time_series[missing_mask] - inpainted_signal[missing_mask]) ** 2)
    # print("MSE Value is: ", mse)

    # Check MSE calculation
    if np.isnan(mse):
        print("MSE calculation resulted in NaN")
        print("Time series at missing mask:", time_series[missing_mask])
        print("Inpainted signal at missing mask:", inpainted_signal[missing_mask])

    return inpainted_signal, mse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].expand(-1, x.size(1), -1)

def create_mask(num_patches, batch_size, mask_ratio=0.15):
    mask = torch.zeros((batch_size, num_patches), dtype=torch.bool)
    num_masked = max(1, int(mask_ratio * num_patches))
    for i in range(batch_size):
        start = torch.randint(0, num_patches - num_masked + 1, (1,))
        mask[i, start:start+num_masked] = True
    return mask

class TransformerMaskedAutoencoder_inference(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, patch_size, dropout=0.1):
        super(TransformerMaskedAutoencoder_inference, self).__init__()
        self.patch_size = patch_size
        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_proj = nn.Linear(d_model, input_dim * patch_size)

    def forward(self, src, tgt):
        batch_size, seq_len, input_dim = src.shape
        num_patches_src = seq_len // self.patch_size

        src = src.reshape(batch_size, num_patches_src, self.patch_size * input_dim)
        tgt = tgt.reshape(batch_size, num_patches_src, self.patch_size * input_dim)
        
        # Project input to d_model dimensions
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)
        
        # Apply positional encoding
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        tgt = self.pos_encoder(tgt.transpose(0, 1)).transpose(0, 1)
        
        # Transpose for transformer layers
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.output_proj(output)
        
        # Reshape output back to original dimensions
        output = output.transpose(0, 1).reshape(batch_size, seq_len, input_dim)
        return output

class TransformerMaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
     num_decoder_layers, dim_feedforward, patch_size, dropout=0.1):
        super(TransformerMaskedAutoencoder, self).__init__()
        self.patch_size = patch_size
        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_proj = nn.Linear(d_model, input_dim * patch_size)

    def forward(self, src, tgt, mask):
        # src and tgt are expected to be of shape (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = src.shape
        num_patches = seq_len // self.patch_size

        # Reshape to (batch_size, num_patches, patch_size * input_dim)
        src = src.reshape(batch_size, num_patches, self.patch_size * input_dim)
        tgt = tgt.reshape(batch_size, num_patches, self.patch_size * input_dim)
        
        # Transpose to (num_patches, batch_size, patch_size * input_dim)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        # Project input to d_model dimensions
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Convert and apply boolean/float mask
        float_mask = mask.float().transpose(0, 1).unsqueeze(-1)
        src = src * (1 - float_mask)
        
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.output_proj(output)

        output = output.transpose(0, 1).reshape(batch_size, seq_len, input_dim)
        return output


def interpolate_missing_values_v4(df, model, device):
    # This is latest used in experiments as of 26/11/2024
    model = model.to(device)
    model.eval()

    time_series = df['time_series'].values
    interp_mask = df['interp_mask'].values.astype(bool)

    available_mask = ~interp_mask
    source_signal = time_series * available_mask
    target_signal = time_series

    source_tensor = torch.from_numpy(source_signal.reshape(1, -1, 1)).float().to(device)
    target_tensor = torch.from_numpy(target_signal.reshape(1, -1, 1)).float().to(device)

    with torch.no_grad():
        reconstructed_signal = model(source_tensor, target_tensor)

    reconstructed_signal = reconstructed_signal.squeeze(0).squeeze(-1).cpu().numpy()

    # Ensure reconstructed signal is valid (check for NaNs)
    if np.isnan(reconstructed_signal).any():
        print("Reconstructed signal contains NaNs!")
        return None, None  # or handle the error more gracefully

    inpainted_signal = time_series.copy()
    inpainted_signal[interp_mask] = reconstructed_signal[interp_mask]

    mse = np.mean((inpainted_signal - time_series ) ** 2)
    return inpainted_signal, mse

def interpolate_missing_values_v5(df, model, device):
    # This is latest used in experiments as of 28/11/2024
    model = model.to(device)
    model.eval()

    time_series = df['time_series'].values
    interp_mask = df['interp_mask'].values.astype(bool)

    # Ensure the signals are 3D (batch_size, sequence_length, input_dim)
    seq_len = len(time_series)
    if seq_len % model.patch_size != 0:
        raise ValueError("Sequence length must be divisible by patch size")

    input_signal = time_series.reshape(1, -1, 1)

    # Reshape mask to match the expected shape by the model
    num_patches = seq_len // model.patch_size
    mask = interp_mask.reshape(num_patches, model.patch_size)
    mask = mask.any(axis=1)  # Reduce to boolean mask per patch
    mask = mask.reshape(1, num_patches)  # Shape: (batch_size, num_patches)

    input_tensor = torch.from_numpy(input_signal).float().to(device)
    mask_tensor = torch.from_numpy(mask).bool().to(device)

    with torch.no_grad():
        reconstructed_signal = model(input_tensor, input_tensor, mask_tensor)

    signal_length = get_effective_signal_length(time_series)
    reconstructed_signal = reconstructed_signal.squeeze(0).squeeze(-1).cpu().numpy()
    reconstructed_signal = reconstructed_signal[:signal_length]
    interp_mask = interp_mask[:signal_length]



    # Ensure reconstructed signal is valid (check for NaNs)
    if np.isnan(reconstructed_signal).any():
        print("Reconstructed signal contains NaNs!")
        return None, None  # or handle the error more gracefully

    inpainted_signal = time_series.copy()
    inpainted_signal = inpainted_signal[:signal_length]

    # print(f"Reconstructed signal shape: {reconstructed_signal.shape}")
    # print(f"Interp mask shape: {interp_mask.shape}")
    # print(f"Inpainted signal shape: {inpainted_signal.shape}")

    inpainted_signal[interp_mask] = reconstructed_signal[interp_mask]

    mse = np.mean((inpainted_signal - time_series[:signal_length]) ** 2)
    return inpainted_signal, mse


def get_effective_signal_length(time_series):
    # Find the last non-zero index
    for i in range(len(time_series) - 1, -1, -1):
        if time_series[i] != 0:
            return i + 1
    return 0


def plot_interpolated_old(ax, x_axis, original_signal, interpolated_signal, missing_mask, color, label):
    ax.plot(x_axis, original_signal, color='blue', linewidth=1, alpha=0.5, label='FHR Original')
    
    # Find ranges of interpolated values
    start_idx = None
    for i in range(len(missing_mask)):
        if missing_mask[i] and start_idx is None:
            start_idx = i
        elif not missing_mask[i] and start_idx is not None:
            if i - start_idx > 1:
                ax.plot(x_axis[start_idx:i], interpolated_signal[start_idx:i], color=color, linewidth=1.5)
            else:
                ax.scatter(x_axis[start_idx:i], interpolated_signal[start_idx:i], color=color, s=8)
            start_idx = None
    if start_idx is not None:
        if len(missing_mask) - start_idx > 1:
            ax.plot(x_axis[start_idx:], interpolated_signal[start_idx:], color=color, linewidth=1.5)
        else:
            ax.scatter(x_axis[start_idx:], interpolated_signal[start_idx:], color=color, s=8)
    
    # Ensure both labels are shown in the legend only once
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_ylim(0, 1)

def plot_interpolated(ax, x_axis, original_signal, interpolated_signal, missing_mask, color, label):
    # Plot original signal
    ax.plot(x_axis[~missing_mask], original_signal[~missing_mask], color='blue', linewidth=1, alpha=0.5, label='FHR Original')
    
    # Find ranges of interpolated values
    interpolated_ranges = []
    start_idx = None
    for i, is_missing in enumerate(missing_mask):
        if is_missing and start_idx is None:
            start_idx = i
        elif not is_missing and start_idx is not None:
            interpolated_ranges.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        interpolated_ranges.append((start_idx, len(missing_mask)))
    
    # Plot interpolated segments
    for start, end in interpolated_ranges:
        if end - start > 1:
            ax.plot(x_axis[start:end], interpolated_signal[start:end], color=color, linewidth=1.5)
        else:
            ax.scatter(x_axis[start:end], interpolated_signal[start:end], color=color, s=8)
    
    # Add label for interpolated signal
    ax.plot([], [], color=color, linewidth=1.5, label=label)
    
    # Ensure both labels are shown in the legend only once
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_ylim(0, 1)


def plot_interpolated_same(ax, x_axis, original_signal, interpolated_signal, missing_mask, color, label):
    # Plot original signal
    # ax.plot(x_axis[~missing_mask], original_signal[~missing_mask], color='blue', linewidth=1, alpha=0.5, label='FHR Original')
    
    # Find ranges of interpolated values
    interpolated_ranges = []
    start_idx = None
    for i, is_missing in enumerate(missing_mask):
        if is_missing and start_idx is None:
            start_idx = i
        elif not is_missing and start_idx is not None:
            interpolated_ranges.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        interpolated_ranges.append((start_idx, len(missing_mask)))
    
    # Plot interpolated segments
    for start, end in interpolated_ranges:
        if end - start > 1:
            ax.plot(x_axis[start:end], interpolated_signal[start:end], color=color, linewidth=1.5)
        else:
            ax.scatter(x_axis[start:end], interpolated_signal[start:end], color=color, s=8)
    
    # Add label for interpolated signal
    ax.plot([], [], color=color, linewidth=1.5, label=label)
    
    # Ensure both labels are shown in the legend only once
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_ylim(0, 1)
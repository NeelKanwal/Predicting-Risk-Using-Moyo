# Author: Neel Kanwal, neel.kanwal0@gmail.com
# Autoeconder training for semi-supervised learning 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 28}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (20, 20)

import seaborn as sns
sns.set_style("white")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from myfunctions import FHRDataset, FHRDataset_v2, collate_fn, FocalLoss, visualize_reconstruction , FHRDataset_encoder,\
 convert_batch_list, make_pretty_cm, TCNModelv2, get_class_weights, weights_to_tensor, VAE, vae_loss, plot_random_signals,\
 FHRDataset_v3, FHRDataset_encoder_v2, vae_loss_v2, VAE_v2
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
from mmcv.cnn import get_model_complexity_info
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd

import time
import pprint

from datetime import datetime
import json
import torchvision
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, brier_score_loss, \
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, precision_score
from scikitplot.metrics import plot_roc, plot_precision_recall, plot_lift_curve, plot_ks_statistic, plot_calibration_curve
from collections import Counter
from torch.cuda.amp import autocast, GradScaler

train_dir = "/......./training/"
val_dir = "/......../validation/"
test_dir = "/......./test/"

input_dim = 7200  # Adjust based on your dataset
hidden_dim = 512
latent_dim = 128
batch_size = 16


num_epochs = 1
early_stopping_patience = 30
  
iterations = 1
opt = [ "SGD", "Adam"]
lr_scheduler = ["ReduceLROnPlateau", "ExponentialLR"] # ExponentialLR
learning_rate = [0.02, 0.002, 0.00002]      

cuda_gpu = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
if torch.cuda.is_available():
    print("Cuda is available")
    # torch.cuda.set_device(cuda_gpu)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{cuda_gpu}" if torch.cuda.is_available() else "cpu")

torch.manual_seed(250)

print("Loading datasets.........")
train_dataset = FHRDataset_encoder_v2(train_dir, sequence_length=input_dim, normalization='minmax')
test_dataset = FHRDataset_v3(test_dir, sequence_length=input_dim, normalization='minmax')

print(f"Length of Training: {len(train_dataset)}, and Test: {len(test_dataset)}")

print("Initializing loaders.........")
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Initialize the model, loss function, and optimizer
for op in opt:
    for sch in lr_scheduler:
        for lr in learning_rate:
            for i in range(iterations):
                print(f"\n//////////////  -----------------  /////////////////\n")    
                print("#########################################################")
                print(f"Optimizer: {op}   Scheduler: {sch}  Learning rate: {lr} ")
                print("#########################################################\n")

                print("Load Model........")
                model = VAE_v2(input_dim, hidden_dim, latent_dim)
                if torch.cuda.is_available():
                    print("Cuda is available")
                    model = model.to(device)

                if op == "SGD":
                    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
                elif op == "Adam":
                    optimizer = Adam(model.parameters(), lr=lr, betas=(0., 0.9), eps=1e-6, weight_decay=0.01)
                else:
                    print("Optimizer does not exists in settings.\n")
                    raise AssertionError

                if sch == "ReduceLROnPlateau":
                    # Reduce learning rate when a metric has stopped improving.
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
                elif sch == "ExponentialLR":
                    # Decays the learning rate of each parameter group by gamma every epoch.
                    scheduler = ExponentialLR(optimizer, gamma=0.8)
                else:
                    print("Scheduler does not exists in settings.\n")
                    raise AssertionError

                pytorch_total_params = sum(p.numel() for p in model.parameters())
                print("Total number of parameters: ", pytorch_total_params)

                print("\nTraining Starts....................")
                now = datetime.now()
                date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
                print(f"\nFiles for will be saved with {date_time} timestamp.")

                if not os.path.exists(os.path.join(os.getcwd(), "experiments", date_time)):
                    if not os.path.exists(os.path.join(os.getcwd(), "experiments")):
                        os.mkdir(os.path.join(os.getcwd(), "experiments"))
                    path = os.path.join(os.getcwd(), "experiments", date_time)
                    os.mkdir(path)
                    print(f"\nDirectory Created {path}.")

                best_val_loss = float('inf')
                epochs_without_improvement = 0
                best_model_weights = {}

                train_losses = []
                val_losses = []
                

                st = time.time()
                for epoch in range(num_epochs):
                    print(f"####### Epoch:{epoch+1} #########")
                    model.train()
                    train_loss, val_loss = 0, 0
                    plot_random_signals(train_loader, num_signals=7, save_path=path)
                    for data, labels in train_loader:
                        if torch.cuda.is_available():
                            data = data.to(device)
                        optimizer.zero_grad()    
                        
                        recon_batch, mu, logvar =  model(data)
                        loss = vae_loss_v2(recon_batch, data, mu, logvar)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        train_loss += loss.item()

                    epoch_tr_loss = train_loss / len(train_loader)
                    print(f"Training loss = {epoch_tr_loss}")
                    train_losses.append(epoch_tr_loss)

                    model.eval()
                    with torch.no_grad():
                        for data, labels in val_loader:
                            if torch.cuda.is_available():
                                data = data.to(device)
                            recon_batch, mu, logvar =  model(data)
                            loss = vae_loss(recon_batch, data, mu, logvar)
                            val_loss += loss.item()

                        epoch_val_loss = val_loss /len(val_loader)
                        print(f"Validation loss = {epoch_val_loss}")
                        val_losses.append(epoch_val_loss)

                    visualize_reconstruction(model, val_loader, device, epoch, save_path=path)

                    # Save the best model weights
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss
                        best_model_weights = model.state_dict()
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if sch == "ReduceLROnPlateau":
                        scheduler.step(epoch_val_loss)
                    elif sch is not None:
                        scheduler.step()

                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"Layer: {name}, Gradient mean: {param.grad.mean()},\
                    #          Gradient std: {param.grad.std()}")
                            
                    # Early stopping
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        break

                seconds = time.time() - st
                minutes = seconds / 60  
                print(f"Total epochs: {epoch + 1}, Training time consumed: {minutes:.2f} minutes")

                plt.clf()
                plt.close('all')
                plt.plot(train_losses, color='b', linewidth=2, label='Training Loss')
                plt.plot(val_losses, color='r', linestyle='--', linewidth=2, label='Validation Loss')

                plt.xlabel('Epoch', fontsize=16)
                plt.ylabel('Loss', fontsize=16)
                plt.title('Training and Validation Losses', fontsize=20)
                plt.legend(fontsize=16)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.savefig(f"{path}/LossCurve.png")
                # plt.show()

                torch.save({'model': best_model_weights}, f"{path}/best_weights.dat")
                # Load the best model weights
                model.load_state_dict(best_model_weights)

                print("\nProgram Completed.........")

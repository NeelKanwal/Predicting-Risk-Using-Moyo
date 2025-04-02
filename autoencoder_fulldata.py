# Author: Neel Kanwal, neel.kanwal0@gmail.com
# Training auto-encoder using full length of FHR, it uses a different version of dataloader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

font = {'family' : 'serif',
        'weight':'normal',
        'size'   : 28}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (24, 16)

import seaborn as sns
sns.set_style("white")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from myfunctions import FHRDataset, FHRDataset_v2, collate_fn, FocalLoss,visualize_reconstruction, calculate_fid, plot_reconstructions, \
 convert_batch_list, make_pretty_cm, get_class_weights, weights_to_tensor, VAE, vae_loss, plot_random_signals, FHRDataset_encoder
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
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

train_dir = "/.../newdata+training/"
val_dir = "/..../validation/"
test_dir = "/...../test/"

input_dim = 7200  # Adjust based on your dataseta
hidden_dim = 1024
latent_dim = 512

batch_size = 256
Norm = 'minmax'  #  'zscore', 'minmax', None
model_name = 'VAE'


num_epochs = 5000
early_stopping_patience = 50
  
iterations = 3
opt = ["Adam"]
lr_scheduler = ["ReduceLROnPlateau", "ExponentialLR"] # ExponentialLR
learning_rate = [ 0.0001]       #0.00001

cuda_gpu = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
if torch.cuda.is_available():
    print("Cuda is available")
    # torch.cuda.set_device(cuda_gpu)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{cuda_gpu}" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1250)

print("Loading datasets.........")
train_dataset = FHRDataset_encoder(train_dir, sequence_length=input_dim, normalization=Norm, force_norm=True) # 'zscore', 'minmax'
val_dataset = FHRDataset_v2(val_dir, sequence_length=input_dim, normalization=Norm, force_norm=True)

print(f"Length of Training: {len(train_dataset)}, and Validation: {len(val_dataset)}")

print("Initializing data loaders.........")
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Initialize the model, loss function, and optimizer
for op in opt:
    for sch in lr_scheduler:
        for lr in learning_rate:
            for i in range(iterations):
                print(f"\n//////////////  -----------------  /////////////////\n")    
                print("#########################################################")
                print(f"Optimizer: {op}   Scheduler: {sch}  Learning rate: {lr} ")
                print("#########################################################\n")

                print("Load Variational AutoEncoder........")
                model = VAE(input_dim, hidden_dim, latent_dim)
                if torch.cuda.is_available():
                    print("Cuda is available")
                    model = model.to(device)

                if op == "SGD":
                    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-5)
                elif op == "Adam":
                    optimizer = Adam(model.parameters(), lr=lr, betas=(0., 0.9), eps=1e-6, weight_decay=1e-5)
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

                print(f"\n{model_name} Training Starts....................")
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
                beta = 0.0

                plot_random_signals(train_loader, model_name, save_path=path, norm=Norm, num_signals=3)

                for epoch in range(num_epochs):
                    print(f"####### Epoch:{epoch+1} #########")
                    model.train()
                    train_loss, val_loss = 0, 0
                    beta = min(1.0, beta + 0.1)  # Gradually increase beta up to 1.0
                    
                    for data, labels in train_loader:
                        if torch.cuda.is_available():
                            data = data.to(device)

                        optimizer.zero_grad()    
                        
                        recon_batch, mu, logvar =  model(data)
                        # print("Input shape:", data.shape)
                        # print("Reconstructed shape:", recon_batch.shape)
                        # print("Mu shape:", mu.shape)
                        # print("Logvar shape:", logvar.shape)

                        loss = vae_loss(recon_batch, data, mu, logvar, beta)
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
                    if epoch % 10 == 0:
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

                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        break

                seconds = time.time() - st
                minutes = seconds / 60  
                print(f"Total epochs: {epoch + 1}, Training time consumed: {minutes:.2f} minutes")

                plt.clf()
                plt.close('all')
                plt.plot(train_losses, color='b', linewidth=4, label='Training Loss')
                plt.plot(val_losses, color='r', linewidth=4,  linestyle='--', label='Validation Loss')

                plt.xlabel('Epoch', fontsize=28)
                plt.ylabel('Loss', fontsize=28)
                plt.title('Training and Validation Losses', fontsize=28)
                plt.legend(fontsize=28)
                plt.xticks(fontsize=28)
                plt.yticks(fontsize=28)
                plt.savefig(f"{path}/LossCurve_VAE.png")
                # plt.show()

                torch.save({'model': best_model_weights}, f"{path}/best_weights.dat")
                # Load the best model weights
                model.load_state_dict(best_model_weights)

                print("-------Validating the model----------")
                model.eval()
                reconstruction_loss = 0
                kl_divergence = 0
                real_features = []
                generated_features = []
                latent_vectors = []
                labels_list = []

                with torch.no_grad():
                    for data, labels in val_loader:
                        if torch.cuda.is_available():
                            data = data.to(device)
                        recon_batch, mu, logvar = model(data)
                        
                        # Reconstruction Loss
                        recon_loss = F.mse_loss(recon_batch.squeeze(-1) , data.squeeze(-1) , reduction='sum')
                        reconstruction_loss += recon_loss.item()
                        
                        # KL Divergence
                        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        kl_divergence += kl_div.item()
                        
                        # Collect features for FID calculation
                        real_features.append(data.cpu().numpy())
                        generated_features.append(recon_batch.cpu().numpy())
                        
                        # Collect latent vectors and labels for visualization
                        latent_vectors.append(mu.cpu().numpy())
                        labels_list.append(labels.cpu().numpy())

                # Calculate average losses
                reconstruction_loss /= len(val_loader.dataset)
                kl_divergence /= len(val_loader.dataset)

                # Calculate FID
                real_features = np.concatenate(real_features)
                generated_features = np.concatenate(generated_features)
                fid_score = calculate_fid(real_features, generated_features)

                print(f"Reconstruction Loss: {reconstruction_loss}")
                print(f"KL Divergence: {kl_divergence}")
                print(f"FID Score: {fid_score}")

                # Latent Space Visualization
                latent_vectors = np.concatenate(latent_vectors)
                labels = np.concatenate(labels_list)

                # Define dimensionality reduction techniques
                reduction_techniques = {
                    'UMAP': umap.UMAP(n_components=2, random_state=42),
                    't-SNE': TSNE(n_components=2, random_state=42),
                    'PCA': PCA(n_components=2, random_state=42)}

                # Define a color map
                color_map = {0: 'blue', 1: 'red', 2: 'green'}
                class_names_outcome30 = {0: 'Normal', 1: 'NICU', 2: 'Death'}
                class_names_resus = {0: 'NO RESUS NEEDED', 1: 'RESUS NEEDED'}

                for technique_name, technique in reduction_techniques.items():
                    # Perform dimensionality reduction
                    latent_reduced = technique.fit_transform(latent_vectors)
                    
                    # Plot for Outcome 30
                    plt.figure(figsize=(12, 12))
                    for label in np.unique(labels[:, 0]):
                        mask = labels[:, 0] == label
                        plt.scatter(latent_reduced[mask, 0], latent_reduced[mask, 1], 
                                    c=color_map[label], label=class_names_outcome30[label], alpha=0.7, s=150)
                    plt.legend(fontsize=18)
                    plt.tight_layout()
                    plt.title(f'{technique_name} Visualization - For Outcome 30')
                    plt.savefig(f'{path}/{technique_name.lower()}_outcome30.png', dpi=300, bbox_inches='tight')
                    plt.close()

                    # Plot for RESUS
                    plt.figure(figsize=(12, 12))
                    for label in np.unique(labels[:, 2]):
                        mask = labels[:, 2] == label
                        plt.scatter(latent_reduced[mask, 0], latent_reduced[mask, 1], 
                                    c=color_map[label], label=class_names_resus[label], alpha=0.6, s=150)
                    plt.legend(fontsize=18)
                    plt.tight_layout()
                    plt.title(f'{technique_name} Visualization - For RESUS')
                    plt.savefig(f'{path}/{technique_name.lower()}_RESUS.png', dpi=300, bbox_inches='tight')
                    plt.close()

                file_names = [os.path.basename(val_loader.dataset.files[i]) for i in range(len(data))]
                plot_reconstructions(real_features, generated_features, file_names, Norm, path, n=5)

print("\nProgram Completed.........")

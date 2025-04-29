# Author: Neel Kanwal (neel.kanwal0@gmail.com)
# Script for training transformer-based masked autoencoder for reconstruction task.
# the trained model weights are later used for inpainting and forecasting applications. 

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
plt.rcParams["figure.figsize"] = (24, 16)
matplotlib.use('Agg')

import seaborn as sns
sns.set_style("white")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from myfunctions import FHRDataset, FHRDataset_v2, collate_fn, FocalLoss,visualize_reconstruction_transformer, calculate_fid, plot_reconstructions, \
 convert_batch_list, make_pretty_cm, get_class_weights, weights_to_tensor, plot_random_signals, FHRDataset_v5_envelope, \
 plot_reconstructions, calculate_fid_v2, calculate_psnr_v2, TransformerMaskedAutoencoder, create_mask, TimeSeriesFrequencyLoss, FHRDataset_encoder 

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
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import math
from focal_frequency_loss import FocalFrequencyLoss as FFL

train_dir = "/nfs/.........../training/"
val_dir = "/nfs/............/validation/"
test_dir = "/nfs/.........../test/"


input_dim = 1  # For univariate time series
d_model = 512 # number of expected features in decoder input

# Multiple attention heads allow the model to focus on different aspects of the input simultaneously.
nhead = 16
num_encoder_layers = 5
num_decoder_layers = 5
dim_feedforward = 1024 # the dimension of the intermediate (hidden) layer 
# in the position-wise feed-forward networks within each transformer layer.

# Increasing d_model affects the entire transformer architecture, 
# while increasing dim_feedforward only affects the feed-forward networks within each layer.

patch_sizes = [30, 60, 120, 240, 480]  # # [60, 120, 240, 480] Assuming time steps per patch

seq_len = 7200
batch_size = 128
Norm = 'minmax'  #  'zscore', 'minmax', None
model_name = 'Transformer_Model_Using_Different_MaskingRatios'

masking_ratios = [0.15] 

sequence_length= 7200

num_epochs = 5000
early_stopping_patience = 20
  
iterations = 2
opt = ["Adam"]
lr_scheduler = ["ReduceLROnPlateau"] # ExponentialLR
learning_rate = [0.0001]    #0.00001

cuda_gpu = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
if torch.cuda.is_available():
    print("Cuda is available")
    # torch.cuda.set_device(cuda_gpu)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{cuda_gpu}" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1250)

print("Loading datasets.........")
# train_dataset = FHRDataset_v5_envelope(train_dir, sequence_length=sequence_length, normalization=Norm, force_norm=True) # 'zscore', 'minmax'
# val_dataset = FHRDataset_v5_envelope(val_dir, sequence_length=sequence_length, normalization=Norm, force_norm=True) # new data is already normalized using minmax
# test_dataset = FHRDataset_v5_envelope(test_dir, sequence_length=sequence_length, normalization=Norm, force_norm=True) 

train_dataset = FHRDataset_encoder(train_dir, sequence_length=sequence_length, normalization=Norm, force_norm=True) # 'zscore', 'minmax'
val_dataset = FHRDataset_v2(val_dir, sequence_length=sequence_length, normalization=Norm, force_norm=True) # new data is already normalized using minmax
test_dataset = FHRDataset_v2(test_dir, sequence_length=sequence_length, normalization=Norm, force_norm=True) 


print(f"Length of Training: {len(train_dataset)}, and Validation: {len(val_dataset)}")

print("Initializing data loaders.........")
train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
for mask_ratio in masking_ratios:
    for patch_size in patch_sizes:
        for op in opt:
            for sch in lr_scheduler:
                for lr in learning_rate:
                    for i in range(iterations):
                        print(f"\n//////////////  -----------------  /////////////////\n")    
                        print("#########################################################")
                        print(f"Optimizer: {op}   Scheduler: {sch}  Learning rate: {lr}  Mask Ratio: {mask_ratio} ")
                        print("#########################################################\n")

                        print(f"Loading Transformer Masked AutoEncoder with patch_size [{patch_size}]........")
                        model = TransformerMaskedAutoencoder(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, patch_size)

                        # Load existing model for second step of training. 
                        # Uncomment to use previous model with patch_size = 30
                        # best_model_wts = '/nfs/student/neel/MoYo_processed_data/experiments/10_25_2024_12_47_52/best_weights.dat'
                        # model.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])

                        if torch.cuda.is_available():
                            print("Cuda is available")
                            model = model.to(device)

                        criterion = nn.MSELoss()
                        freq_loss_fn = TimeSeriesFrequencyLoss(alpha=1.0)

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
                        learning_rates = []
                        

                        st = time.time()
                       # plot_random_signals(train_loader,  model_name, save_path=path, norm=Norm, num_signals=2)

                        #data_index = 5 # 5 for hybrid-30 and 6 for hybrid-60 and 1 for interpolated data
                        model.train()
                        for epoch in range(num_epochs):
                            print(f"####### Epoch:{epoch+1} #########")
                            train_loss, val_loss = 0, 0
                            tot_loss = 0
                            for data, labels in train_loader:
                                if torch.cuda.is_available():
                                    data = data.to(device)
                                
                                if data.dim() == 2:
                                    data = data.unsqueeze(-1)
                                elif data.dim() == 3 and data.size(1) == 1 and data.size(2) == 1:
                                    data = data.squeeze(1)  # Remove the extra dimension if present

                                # Ensure the shape is correct
                                assert data.shape[1] == 7200 and data.shape[2] == 1, f"Unexpected data shape: {data.shape}"
                                
                                # Get dimensions
                                batch_size, seq_len, input_dim = data.shape
                                # print(f"Data shape: {data.shape}")
                                
                                # Calculate number of patches
                                num_patches = seq_len // model.patch_size
                                # print(f"Number of patches: {num_patches}")
                                
                                # Create continuous mask for patches
                                mask = create_mask(num_patches, batch_size, mask_ratio=mask_ratio)
                                # should create a mask for patches, not individual time steps.
                                mask = mask.to(device)
                                # print(f"Mask shape: {mask.shape}")

                                # Expand mask to match data dimensions
                                expanded_mask = mask.repeat(1, 1, model.patch_size)
                                # print(f"Expanded mask shape before view: {expanded_mask.shape}")
                                expanded_mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size).view(batch_size, seq_len, input_dim)
                                # print(f"Expanded mask shape after view: {expanded_mask.shape}")

                               # Forward pass
                                output = model(data, data, mask)
                                # print(f"Output shape: {output.shape}")
                                
                                # Compute loss
                                loss = criterion(output[expanded_mask], data[expanded_mask])
                                freq_loss = freq_loss_fn(output, data)
                                # print("Criterion Loss, ", loss)
                                # print("Frequency Loss, ", freq_loss)

                                tot_loss = 0.95 * loss + 0.05 * freq_loss
                                # You might want to focus on masked regions only
                                
                                # Backward pass and optimize
                                optimizer.zero_grad()    
                                tot_loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                train_loss += tot_loss.item()

                            epoch_tr_loss = train_loss / len(train_loader)
                            print(f"Training loss = {epoch_tr_loss}")
                            train_losses.append(epoch_tr_loss)
                            tot_loss = 0

                            current_lr = optimizer.param_groups[0]['lr']
                            learning_rates.append(current_lr)

                            model.eval()
                            with torch.no_grad():
                                for data, labels in val_loader:
                                    if torch.cuda.is_available():
                                        data = data.to(device)

                                    if data.dim() == 2:
                                        data = data.unsqueeze(-1)
                                
                                   # Get dimensions
                                    batch_size, seq_len, input_dim = data.shape
                                    
                                    # Calculate number of patches
                                    num_patches = seq_len // model.patch_size
                                    
                                    # Create continuous mask for patches
                                    mask = create_mask(num_patches, batch_size, mask_ratio=mask_ratio)
                                    mask = mask.to(device)

                                    # Expand mask to match data dimensions
                                    expanded_mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size * input_dim)
                                    expanded_mask = expanded_mask.view(batch_size, seq_len, input_dim)

                                   # Forward pass
                                    output = model(data, data, mask)
                                    
                                    # Compute loss      # focus on masked regions only
                                    loss = criterion(output[expanded_mask], data[expanded_mask])
                                    freq_loss = freq_loss_fn(output, data)

                                    tot_loss = 0.95 * loss + 0.05 * freq_loss
                               

                                    val_loss += tot_loss.item()

                                epoch_val_loss = val_loss /len(val_loader)
                                print(f"Validation loss = {epoch_val_loss}")
                                val_losses.append(epoch_val_loss)

                            
                            if epoch % 10 == 0:
                                visualize_reconstruction_transformer(model, val_loader, device, epoch, save_path=path)
                            

                            # Save the best model weights
                            if epoch_val_loss < best_val_loss:
                                best_val_loss = epoch_val_loss
                                best_model_weights = model.state_dict()
                                epochs_without_improvement = 0
                            else:
                                epochs_without_improvement += 1

                                # uncomment to use lr scheduler
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

                        plt.xlabel('Epoch', fontsize=20)
                        plt.ylabel('Loss', fontsize=20)
                        plt.title('Training and Validation Losses', fontsize=24)
                        plt.legend(fontsize=20)
                        plt.xticks(fontsize=20)
                        plt.yticks(fontsize=20)
                        plt.savefig(f"{path}/LossCurve_TransformerAE.png")
                        # plt.show()

                        plt.figure(figsize=(15, 6))
                        plt.plot(range(1, len(learning_rates) + 1), learning_rates, marker='o')
                        plt.xlabel('Epoch', fontsize=14)
                        plt.ylabel('Learning Rate', fontsize=14)
                        plt.title('Learning Rate vs. Epoch', fontsize=16)
                        plt.grid(True)
                        plt.savefig(f"{path}/LearningRateCurve.png")
                        plt.close()

                        torch.save({'model': best_model_weights}, f"{path}/best_weights.dat")
                        # Load the best model weights
                        model.load_state_dict(best_model_weights)

                        print(f"\n-------Validating the model with patch size [{patch_size}] with masking ratio [{mask_ratio}]----------")
                        model.eval()
                        reconstruction_loss = 0

                        real_features = []
                        generated_features = []
                        labels_list = []

                        with torch.no_grad():
                            for data, labels in val_loader:
                                if torch.cuda.is_available():
                                    data = data.to(device)
                                
                                if data.dim() == 2:
                                    data = data.unsqueeze(-1)
                                
                                # Get dimensions
                                batch_size, seq_len, input_dim = data.shape
                                
                                # Calculate number of patches
                                num_patches = seq_len // model.patch_size
                                
                                # Create continuous mask for patches
                                mask = create_mask(num_patches, batch_size, mask_ratio=mask_ratio)
                                mask = mask.to(device)
                                
                                # Forward pass
                                output = model(data, data, mask)
                                
                                # Reshape mask to match data dimensions
                                # This is necessary because the mask is created at the patch level, 
                                #but we need to apply it to the full sequence for loss calculation.
                                expanded_mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size * input_dim)
                                expanded_mask = expanded_mask.view(batch_size, seq_len, input_dim)
                                
                                # Reconstruction Loss (only on masked positions)
                                recon_loss = F.mse_loss(output[expanded_mask], data[expanded_mask], reduction='sum')
                                reconstruction_loss += recon_loss.item()
                                
                                # Collect features for visualization
                                real_features.append(data.cpu().numpy())
                                generated_features.append(output.cpu().numpy())
                                
                                # Collect labels for visualization
                                labels_list.append(labels.cpu().numpy())

                        # Calculate average loss
                        reconstruction_loss /= len(val_loader.dataset)

                        print(f"Reconstruction Loss: {reconstruction_loss}")

                        # Prepare data for visualization
                        real_features = np.concatenate([f.reshape(f.shape[0], -1) for f in real_features])
                        generated_features = np.concatenate([f.reshape(f.shape[0], -1) for f in generated_features])
                        labels = np.concatenate(labels_list)

                        ssim_score = ssim(real_features, generated_features, data_range=generated_features.max() - generated_features.min())
                        fid_score = calculate_fid_v2(real_features, generated_features)
                        psnr = calculate_psnr_v2(real_features, generated_features)
                        print(f"Peak Signal-to-Noise Ratio: {psnr}")
                        print(f"Structural Similarity Index: {ssim_score}")
                        print(f"Fréchet Inception Distance: {fid_score}")

                        mse = np.mean((real_features - generated_features)**2)
                        rmse = np.sqrt(mse)
                        mae = np.mean(np.abs(real_features - generated_features))

                        # For correlation, we'll flatten the arrays
                        real_flat = real_features.flatten()
                        gen_flat = generated_features.flatten()
                        correlation = np.corrcoef(real_flat, gen_flat)[0, 1]

                        # print(f"Structural Similarity Index: {ssim_score}")
                        print(f"Mean Squared Error: {mse}")
                        print(f"Root Mean Squared Error: {rmse}")
                        print(f"Mean Absolute Error: {mae}")
                        print(f"Correlation Coefficient: {correlation}")


                        # Define dimensionality reduction techniques
                        # reduction_techniques = {'UMAP': umap.UMAP(n_components=2, random_state=42),
                        #     't-SNE': TSNE(n_components=2, random_state=42),
                        #     'PCA': PCA(n_components=2, random_state=42)}

                        # # Define a color map
                        # color_map = {0: 'blue', 1: 'red', 2: 'green'}
                        # class_names_outcome30 = {0: 'Normal', 1: 'NICU', 2: 'Death'}
                        # class_names_resus = {0: 'NO RESUS NEEDED', 1: 'RESUS NEEDED'}

                        # for technique_name, technique in reduction_techniques.items():
                        #     # Perform dimensionality reduction on the generated features
                        #     features_reduced = technique.fit_transform(generated_features)
                            
                        #     # Plot for Outcome 30
                        #     plt.figure(figsize=(12, 12))
                        #     for label in np.unique(labels[:, 0]):
                        #         mask = labels[:, 0] == label
                        #         plt.scatter(features_reduced[mask, 0], features_reduced[mask, 1], 
                        #                     c=color_map[label], label=class_names_outcome30[label], alpha=0.7, s=150)
                        #     plt.legend(fontsize=18)
                        #     plt.tight_layout()
                        #     plt.title(f'{technique_name} Visualization - For Outcome 30')
                        #     plt.savefig(f'{path}/{technique_name.lower()}_outcome30.png', dpi=300, bbox_inches='tight')
                        #     plt.close()

                        #     # Plot for RESUS
                        #     plt.figure(figsize=(12, 12))
                        #     for label in np.unique(labels[:, 2]):
                        #         mask = labels[:, 2] == label
                        #         plt.scatter(features_reduced[mask, 0], features_reduced[mask, 1], 
                        #                     c=color_map[label], label=class_names_resus[label], alpha=0.6, s=150)
                        #     plt.legend(fontsize=18)
                        #     plt.tight_layout()
                        #     plt.title(f'{technique_name} Visualization - For RESUS')
                        #     plt.savefig(f'{path}/{technique_name.lower()}_RESUS.png', dpi=300, bbox_inches='tight')
                        #     plt.close()

                        # Plot reconstructions
                        file_names = [os.path.basename(val_loader.dataset.files[i]) for i in range(len(data))]
                        # plot_reconstructions(real_features, generated_features, file_names, 'Masked',  path, n=5)


                        print(f"\n-------Testing the model with patch size [{patch_size}] with masking ratio [{mask_ratio}]----------")
                        model.eval()
                        reconstruction_loss = 0

                        real_features = []
                        generated_features = []
                        labels_list = []

                        with torch.no_grad():
                            for data, labels in test_loader:
                                if torch.cuda.is_available():
                                    data = data.to(device)
                                
                                if data.dim() == 2:
                                    data = data.unsqueeze(-1)
                                
                                # Get dimensions
                                batch_size, seq_len, input_dim = data.shape
                                
                                # Calculate number of patches
                                num_patches = seq_len // model.patch_size
                                
                                # Create continuous mask for patches
                                mask = create_mask(num_patches, batch_size, mask_ratio=mask_ratio)
                                mask = mask.to(device)
                                
                                # Forward pass
                                output = model(data, data, mask)
                                
                                # Reshape mask to match data dimensions
                                # This is necessary because the mask is created at the patch level, 
                                #but we need to apply it to the full sequence for loss calculation.
                                expanded_mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size * input_dim)
                                expanded_mask = expanded_mask.view(batch_size, seq_len, input_dim)
                                
                                # Reconstruction Loss (only on masked positions)
                                recon_loss = F.mse_loss(output[expanded_mask], data[expanded_mask], reduction='sum')
                                reconstruction_loss += recon_loss.item()
                                
                                # Collect features for visualization
                                real_features.append(data.cpu().numpy())
                                generated_features.append(output.cpu().numpy())
                                
                                # Collect labels for visualization
                                labels_list.append(labels.cpu().numpy())

                        # Calculate average loss
                        reconstruction_loss /= len(test_loader.dataset)

                        print(f"Reconstruction Loss: {reconstruction_loss}")

                        # Prepare data for visualization
                        real_features = np.concatenate([f.reshape(f.shape[0], -1) for f in real_features])
                        generated_features = np.concatenate([f.reshape(f.shape[0], -1) for f in generated_features])
                        labels = np.concatenate(labels_list)

                        ssim_score = ssim(real_features, generated_features, data_range=generated_features.max() - generated_features.min())
                        fid_score = calculate_fid_v2(real_features, generated_features)
                        psnr = calculate_psnr_v2(real_features, generated_features)
                        print(f"Peak Signal-to-Noise Ratio: {psnr}")
                        print(f"Structural Similarity Index: {ssim_score}")
                        print(f"Fréchet Inception Distance: {fid_score}")

                        mse = np.mean((real_features - generated_features)**2)
                        rmse = np.sqrt(mse)
                        mae = np.mean(np.abs(real_features - generated_features))

                        # For correlation, we'll flatten the arrays
                        real_flat = real_features.flatten()
                        gen_flat = generated_features.flatten()
                        correlation = np.corrcoef(real_flat, gen_flat)[0, 1]

                        # print(f"Structural Similarity Index: {ssim_score}")
                        print(f"Mean Squared Error: {mse}")
                        print(f"Root Mean Squared Error: {rmse}")
                        print(f"Mean Absolute Error: {mae}")
                        print(f"Correlation Coefficient: {correlation}")


                        print("\nProgram Completed.........")

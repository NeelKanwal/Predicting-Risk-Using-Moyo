# Author: Neel Kanwal, neel.kanwal0@gmail.com
# This is transformer-based classification model for outcome classification training. It uses new version of dataset and loads a different version of dataloader.
# this uses three input for training. Multivariate training.


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
from myfunctions import  collate_fn, convert_batch_list, make_pretty_cm, get_class_weights, weights_to_tensor, FHRDataset_v6_envelope,\
   TransformerClassifier_step32,  tversky_loss

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

train_dir = "/nfs/student/neel/MoYo_processed_data/labelled_data_with_envelope/interp_training/"
val_dir = "/nfs/student/neel/MoYo_processed_data/labelled_data_with_envelope/interp_validation/"
test_dir = "/nfs/student/neel/MoYo_processed_data/labelled_data_with_envelope/interp_test/"

input_dim = 13 # For univariate time series set to 1
d_model = 512 # number of expected features in decoder input

# Multiple attention heads allow the model to focus on different aspects of the input simultaneously.
nhead = 16
num_encoder_layers = 5
dim_feedforward = 1024 # the dimension of the intermediate (hidden) layer 
# in the position-wise feed-forward networks within each transformer layer.
dropout = 0.1  # Valid dropout value

# Increasing d_model affects the entire transformer architecture, 
# while increasing dim_feedforward only affects the feed-forward networks within each layer.

patch_sizes = [30]  # # [60, 120, 240, 480] Assuming time steps per patch

batch_size = 256
model_name = 'Transformer_2classes_NewData_model1_scheduler_multichannel'

sequence_length= 7200

num_epochs = 1
early_stopping_patience = 50
  
iterations = 1
opt = ["Adam", "SGD"]
lr_scheduler = ["ReduceLROnPlateau"] # ExponentialLR
learning_rate = [0.01, 0.0001]    #0.00001

cuda_gpu = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
if torch.cuda.is_available():
    print("Cuda is available")
    # torch.cuda.set_device(cuda_gpu)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{cuda_gpu}" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1250)

print("Loading datasets.........")
train_dataset = FHRDataset_v6_envelope(train_dir, sequence_length=sequence_length) # 'zscore', 'minmax'
val_dataset = FHRDataset_v6_envelope(val_dir, sequence_length=sequence_length) # new data is already normalized using minmax
test_dataset = FHRDataset_v6_envelope(test_dir, sequence_length=sequence_length) # new data is already normalized using minmax


print(f"Length of Training: {len(train_dataset)}, Validation: {len(val_dataset)} and Test: {len(test_dataset)}.")

print("Initializing data loaders.........")
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# Initialize the model, loss function, and optimizer
for patch_size in patch_sizes:
    for op in opt:
        for sch in lr_scheduler:
            for lr in learning_rate:
                for i in range(iterations):
                    print(f"\n//////////////  -----------------  /////////////////\n")    
                    print("#########################################################")
                    print(f"Optimizer: {op}   Scheduler: {sch}  Learning rate: {lr} ")
                    print("#########################################################\n")

                    print(f"Loading Transformer Masked AutoEncoder with patch_size [{patch_size}]........")
                    model = TransformerClassifier_step32(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, patch_size, dropout)
                 
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
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
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

                    train_losses_30min, val_losses_30min = [], []
                    train_losses_resus, val_losses_resus = [], []

                    train_losses = []
                    val_losses = []

                    st = time.time()
                    
                    # plot_random_signals_v2(train_loader,  model_name, save_path=path, norm='MinMax', num_signals=2)

                    #data_index = 5 # 5 for hybrid-30 and 6 for hybrid-60 and 1 for interpolated data
                    for epoch in range(num_epochs):
                        print(f"####### Epoch:{epoch+1} #########")
                        train_loss_30min, train_loss_resus = 0.0, 0.0 
                        val_loss_30min, val_loss_resus = 0.0, 0.0
                        total_train_loss, total_val_loss = 0.0, 0.0

                        # Training
                        model.train()
                        correct_30min, correct_resus = 0, 0
                        total_30min, total_resus = 0, 0

                        for data, mask, labels_30min, labels_resus in train_loader:
                            if torch.cuda.is_available():
                                data, mask, labels_30min, labels_resus  = data.to(device), mask.to(device), labels_30min.to(device), labels_resus.to(device)
                            print(f"Data shape: {data.shape}")
                            optimizer.zero_grad()

                            # Forward pass
                            outputs_resus, outputs_outcome = model(data)

                            # Calculate Tversky loss for resuscitation and outcome
                            loss_resus = tversky_loss(torch.softmax(outputs_resus, dim=1), F.one_hot(labels_resus, num_classes=2))
                            loss_outcome = tversky_loss(torch.softmax(outputs_outcome, dim=1), F.one_hot(labels_30min, num_classes=3))

                            # Combine losses
                            loss = loss_resus + loss_outcome

                            loss.backward()

                            if sch == "ReduceLROnPlateau":
                                scheduler.step(loss)
                            else:
                                scheduler.step()

                            train_loss_resus += loss_resus.item() * data.size(0)
                            train_loss_30min += loss_outcome.item() * data.size(0)  # Note: Using loss_outcome for labels_30min

                            # Calculate accuracy
                            _, predicted_resus = torch.max(outputs_resus.data, 1)
                            _, predicted_outcome = torch.max(outputs_outcome.data, 1)

                            total_resus += labels_resus.size(0)
                            total_30min += labels_30min.size(0)

                            correct_resus += (predicted_resus == labels_resus).sum().item()
                            correct_30min += (predicted_outcome == labels_30min).sum().item()

                        train_loss_resus /= len(train_dataset)
                        train_loss_30min /= len(train_dataset)

                        total_train_loss = train_loss_resus + train_loss_30min

                        train_losses_resus.append(train_loss_resus)
                        train_losses_30min.append(train_loss_30min)
                        train_losses.append(total_train_loss)

                        seconds = time.time() - st
                        minutes = seconds / 60

                        train_acc_resus = 100 * correct_resus / total_resus
                        train_acc_30min = 100 * correct_30min / total_30min

                        # print(f"Epoch {epoch+1}, Train Loss Resus: {train_loss_resus:.4f}, Train Loss Outcome: {train_loss_30min:.4f}")
                        # print(f"Epoch {epoch+1}, Train Acc Resus: {train_acc_resus:.2f}%, Train Acc Outcome: {train_acc_30min:.2f}%")

                        # Validation
                        model.eval()

                        val_correct_resus = 0
                        val_correct_outcome = 0
                        val_total_resus = 0
                        val_total_outcome = 0

                        with torch.no_grad():
                            for data, mask, labels_30min, labels_resus in val_loader:
                                if torch.cuda.is_available():
                                    data, mask, labels_30min, labels_resus  = data.to(device), mask.to(device), labels_30min.to(device), labels_resus.to(device)

                                outputs_resus, outputs_outcome = model(data)

                                loss_resus = tversky_loss(torch.softmax(outputs_resus, dim=1), F.one_hot(labels_resus, num_classes=2))
                                loss_outcome = tversky_loss(torch.softmax(outputs_outcome, dim=1), F.one_hot(labels_30min, num_classes=3))

                                val_loss_resus += loss_resus.item() * data.size(0)
                                val_loss_30min += loss_outcome.item() * data.size(0)

                                _, predicted_resus = torch.max(outputs_resus.data, 1)
                                _, predicted_outcome = torch.max(outputs_outcome.data, 1)

                                val_total_resus += labels_resus.size(0)
                                val_total_outcome += labels_30min.size(0)

                                val_correct_resus += (predicted_resus == labels_resus).sum().item()
                                val_correct_outcome += (predicted_outcome == labels_30min).sum().item()

                        val_loss_resus /= len(val_dataset)
                        val_loss_30min /= len(val_dataset)

                        total_val_loss = val_loss_resus + val_loss_30min

                        val_losses_resus.append(val_loss_resus)
                        val_losses_30min.append(val_loss_30min)
                        val_losses.append(total_val_loss)

                        val_acc_resus = 100 * val_correct_resus / val_total_resus
                        val_acc_outcome = 100 * val_correct_outcome / val_total_outcome

                        print(f"Epoch {epoch+1}, Val Loss Resus: {val_loss_resus:.4f}, Val Loss Outcome: {val_loss_30min:.4f}")
                        print(f"Epoch {epoch+1}, Val Acc Resus: {val_acc_resus:.2f}%, Val Acc Outcome: {val_acc_outcome:.2f}%")
                        print(f"\nTotal validation loss = {total_val_loss}")

                        epoch_val_loss = total_val_loss
                                            
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

                    # Plot the training and validation losses
                    plt.figure(figsize=(20, 10))
                    plt.plot(train_losses_30min, color='r', linewidth=2, label='Training Loss - 30 min')
                    plt.plot(train_losses_resus, color='r', linestyle='--', linewidth=2, label='Training Loss - Resus')

                    plt.plot(val_losses_30min, color='b', linewidth=2, label='Validation Loss - 30m')
                    plt.plot(val_losses_resus, color='b', linestyle='--', linewidth=2, label='Validation Loss - Resus')

                    val_loss = [val_losses_30min[i] + val_losses_resus[i] for i in range(len(val_losses_30min))]
                    plt.plot(val_loss, color='k', linestyle='-.', linewidth=3, label='Validation Loss - Total')

                    train_loss = [train_losses_30min[i] + train_losses_resus[i] for i in range(len(train_losses_30min))]
                    plt.plot(train_loss, color='g', linestyle='-.', linewidth=3, label='Training Loss - Total')

                    plt.xlabel('Epoch', fontsize=16)
                    plt.ylabel('Loss', fontsize=16)
                    plt.title('Training and Validation Losses', fontsize=20)
                    plt.legend(fontsize=16)
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)
                    plt.savefig(f"{path}/LossCurve-2Classifiers_OneChannel.png")
                    # plt.show()

                    torch.save({'model': best_model_weights}, f"{path}/best_weights.dat")
                    # Load the best model weights
                    model.load_state_dict(best_model_weights)


                    print("Validation the model.........")
                    # Test the model
                    model.eval()

                    correct_30min = 0
                    correct_24hours = 0
                    correct_resus = 0

                    total_30min = 0
                    total_24hours = 0
                    total_resus = 0


                    test_acc_30min = 0
                    test_acc_24hours = 0
                    total_acc_resus = 0

                    preds_30min, probs_30min,  label_30min = [], [], []
                    preds_resus, probs_resus, label_resus = [], [], []

                    with torch.no_grad():
                        for data, mask, labels_30min, labels_resus in val_loader:
                            if torch.cuda.is_available():
                                data, mask, labels_30min, labels_resus  = data.to(device), mask.to(device), labels_30.to(device), labels_resus.to(device)

                            outputs_resus, outputs_outcome = model(data)

                            probabilities_outputs_30min = F.softmax(outputs_30min, dim=1)
                            probabilities_outputs_resus = F.softmax(outputs_resus, dim=1)
            

                            # Calculate accuracy
                            _, predicted_30min = torch.max(outputs_30min.data, 1)
                            _, predicted_resus = torch.max(outputs_resus.data, 1)

                            total_30min += labels_30min.size(0)
                            total_resus += labels_resus.size(0)


                            correct_30min += (predicted_30min == labels_30min).sum().item()
                            correct_resus += (predicted_resus == labels_resus).sum().item()

                            label_30min.append(list(labels_30min.cpu().numpy()))
                            label_resus.append(list(labels_resus.cpu().numpy()))

                            preds_30min.append(list(predicted_30min.cpu().numpy()))
                            preds_resus.append(list(predicted_resus.cpu().numpy()))

                            probs_30min.append(list(np.around(probabilities_outputs_30min.detach().cpu().numpy(), decimals=4)))
                            probs_resus.append(list(np.around(probabilities_outputs_resus.detach().cpu().numpy(), decimals=4)))

                    label_30min, preds_30min, probs_30min = convert_batch_list(label_30min), convert_batch_list(preds_30min), convert_batch_list(probs_30min)
                    label_resus, preds_resus, probs_resus = convert_batch_list(label_resus), convert_batch_list(preds_resus), convert_batch_list(probs_resus)


                    label_30min = [x+1 for x in label_30min]
                    preds_30min = [x+1 for x in preds_30min]


                    file_names = [im.split(".")[0] for im in val_loader.dataset.files]
                    data = {"files": file_names, "ground_truth_30min": label_30min,  "predicted_30min": preds_30min, "prob_30min": probs_30min, \
                        "ground_truth_resus": label_resus,"predicted_resus": preds_resus, "prob_resus": probs_resus}

                    dframe = pd.DataFrame(data)
                    print("Length of Validation data ", len(dframe))
                    with pd.ExcelWriter(f"{path}/Validation_2classes._multichannelxlsx") as wr:
                            dframe.to_excel(wr, index=False)

                    test_acc_30min = 100 * correct_30min / total_30min
                    test_acc_resus = 100 * correct_resus / total_resus

                    print(f'Valid-Accuracy 30min: {test_acc_30min:.2f}%,  Valid-Accuracy Resus: {test_acc_resus:.2f}%')

                    cm = confusion_matrix(label_30min, preds_30min)
                    make_pretty_cm(cm, group_names=['0','1','2'], figsize=(8,8), title= f"{path}/Valid-30min-CM")

                    cm = confusion_matrix(label_resus, preds_resus)
                    make_pretty_cm(cm, group_names=['0','1'], figsize=(8,8), title= f"{path}/Valid-RESUS-CM")

                    print("\nTesting the model.........")
                    # Test the model

                    correct_30min = 0
                    correct_24hours = 0
                    correct_resus = 0

                    total_30min = 0
                    total_24hours = 0
                    total_resus = 0


                    test_acc_30min = 0
                    test_acc_24hours = 0
                    total_acc_resus = 0

                    preds_30min, probs_30min,  label_30min = [], [], []
                    preds_resus, probs_resus, label_resus = [], [], []

                    with torch.no_grad():
                        for data,mask, labels_30min, labels_resus  in test_loader:
                            if torch.cuda.is_available():
                                data, mask, labels_30min, labels_resus  = data.to(device), mask.to(device), labels_30min.to(device), labels_resus.to(device)
                    

                            outputs_resus, outputs_outcome = model(data)

                            probabilities_outputs_30min = F.softmax(outputs_30min, dim=1)
                            probabilities_outputs_resus = F.softmax(outputs_resus, dim=1)
            

                            # Calculate accuracy
                            _, predicted_30min = torch.max(outputs_30min.data, 1)
                            _, predicted_resus = torch.max(outputs_resus.data, 1)

                            total_30min += labels_30min.size(0)
                            total_resus += labels_resus.size(0)


                            correct_30min += (predicted_30min == labels_30min).sum().item()
                            correct_resus += (predicted_resus == labels_resus).sum().item()

                            label_30min.append(list(labels_30min.cpu().numpy()))
                            label_resus.append(list(labels_resus.cpu().numpy()))

                            preds_30min.append(list(predicted_30min.cpu().numpy()))
                            preds_resus.append(list(predicted_resus.cpu().numpy()))

                            probs_30min.append(list(np.around(probabilities_outputs_30min.detach().cpu().numpy(), decimals=4)))
                            probs_resus.append(list(np.around(probabilities_outputs_resus.detach().cpu().numpy(), decimals=4)))

                    label_30min, preds_30min, probs_30min = convert_batch_list(label_30min), convert_batch_list(preds_30min), convert_batch_list(probs_30min)
                    label_resus, preds_resus, probs_resus = convert_batch_list(label_resus), convert_batch_list(preds_resus), convert_batch_list(probs_resus)


                    label_30min = [x+1 for x in label_30min]
                    preds_30min = [x+1 for x in preds_30min]


                    file_names = [im.split(".")[0] for im in test_loader.dataset.files]
                    data = {"files": file_names, "ground_truth_30min": label_30min,  "predicted_30min": preds_30min, "prob_30min": probs_30min, \
                        "ground_truth_resus": label_resus,"predicted_resus": preds_resus, "prob_resus": probs_resus}

                    dframe = pd.DataFrame(data)
                    print("Length of Test data ", len(dframe))
                    with pd.ExcelWriter(f"{path}/test_2classes_multichannel.xlsx") as wr:
                            dframe.to_excel(wr, index=False)

                    test_acc_30min = 100 * correct_30min / total_30min
                    test_acc_resus = 100 * correct_resus / total_resus

                    print(f'test-Accuracy 30min: {test_acc_30min:.2f}%,  test-Accuracy Resus: {test_acc_resus:.2f}%')

                    cm = confusion_matrix(label_30min, preds_30min)
                    make_pretty_cm(cm, group_names=['0','1','2'], figsize=(8,8), title= f"{path}/test-30min-CM")

                    cm = confusion_matrix(label_resus, preds_resus)
                    make_pretty_cm(cm, group_names=['0','1'], figsize=(8,8), title= f"{path}/test-RESUS-CM")


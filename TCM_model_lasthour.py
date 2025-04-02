## Author: Neel Kanwal, neel.kanwal0@gmail.com
#This script uses a temporal convolutional neural network to a univariate (FHR) time series, use its last hour (7200 samples)
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
from myfunctions import FHRDataset, FHRDataset_v2, TCNModel, collate_fn, FocalLoss,\
 convert_batch_list, make_pretty_cm, TCNModelv2, get_class_weights, weights_to_tensor
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils
from mmcv.cnn import get_model_complexity_info
import time
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch.nn.functional as F

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
from scikitplot.metrics import plot_roc, plot_precision_recall, plot_lift_curve, plot_ks_statistic, \
    plot_calibration_curve
from collections import Counter

train_dir = "/..../training/"
val_dir = "/..../validation/"
test_dir = "/..../test/"

input_size = 1
num_classes_30min = 3
num_classes_24hours = 3
num_classes_resus = 2
num_channels = 64
kernel_size = 6
batch_size = 256
num_epochs = 100
early_stopping_patience = 30
  
iterations = 1
opt = [ "SGD", "Adam"]
lr_scheduler = ["ReduceLROnPlateau"] # ExponentialLR
learning_rate = [0.001, 0.00001]      

cuda_gpu = 5
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
if torch.cuda.is_available():
    print("Cuda is available")
    # torch.cuda.set_device(cuda_gpu)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{cuda_gpu}" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1250)

print("Loading datasets.........")
train_dataset = FHRDataset_v2(train_dir, sequence_length=7200, normalization='zscore')
val_dataset = FHRDataset_v2(val_dir, sequence_length=7200, normalization='zscore')
test_dataset = FHRDataset_v2(test_dir, sequence_length=7200, normalization='zscore')

print(f"Length of Training: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

print("Initializing loaders.........")
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# Get class weights for each task
class_weights_30min = get_class_weights(train_loader, task_idx=0)
class_weights_24hours = get_class_weights(train_loader, task_idx=1)
class_weights_resus = get_class_weights(train_loader, task_idx=2)

print("Class weights for 30-minute outcome:", class_weights_30min)
print("Class weights for 24-hour outcome:", class_weights_24hours)
print("Class weights for resuscitation outcome:", class_weights_resus)

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
                model = TCNModelv2(input_size, num_classes_30min, num_classes_24hours, num_classes_resus,\
                 num_channels, kernel_size)
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
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
                elif sch == "ExponentialLR":
                    # Decays the learning rate of each parameter group by gamma every epoch.
                    scheduler = ExponentialLR(optimizer, gamma=0.8)
                else:
                    print("Scheduler does not exists in settings.\n")
                    raise AssertionError

                pytorch_total_params = sum(p.numel() for p in model.parameters())
                print("Total number of parameters: ", pytorch_total_params)

                weights_30min = weights_to_tensor(class_weights_30min, num_classes_30min).to(device)
                weights_24hours = weights_to_tensor(class_weights_24hours, num_classes_24hours).to(device)
                weights_resus = weights_to_tensor(class_weights_resus, num_classes_resus).to(device)

                # Initialize the loss functions with class weights
                criterion_30min = FocalLoss(weight=weights_30min)
                criterion_24hour = FocalLoss(weight=weights_24hours)
                criterion_resus = FocalLoss(weight=weights_resus)

                # Training loop
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
                train_losses_24hours, val_losses_24hours = [], []
                train_losses_resus, val_losses_resus = [], []

                train_losses = []
                val_losses = []

                st = time.time()

                for epoch in range(num_epochs):
                    print(f"####### Epoch:{epoch+1} #########")
                    train_loss_30min = 0.0
                    train_loss_24hours = 0.0
                    train_loss_resus = 0.0
                    val_loss_30min = 0.0
                    val_loss_24hours = 0.0
                    val_loss_resus = 0.0
                    total_train_loss = 0.0
                    total_val_loss = 0.0

                    # Training
                    model.train()
                    correct_30min = 0
                    correct_24hours = 0
                    correct_resus = 0
                    total_30min = 0
                    total_24hours = 0
                    total_resus = 0

                    for data, labels in train_loader:
                        if torch.cuda.is_available():
                            # data, target = data.cuda(), labels.cuda()
                            data, labels = data.to(device), labels.to(device)

                        labels_30min, labels_24hours, labels_resus = labels[:, 0], labels[:, 1], labels[:, 2]
                        optimizer.zero_grad()
                        outputs_30min, outputs_24hours, outputs_resus = model(data)

                        loss_30min = criterion_30min(outputs_30min, labels_30min)
                        loss_24hours = criterion_24hour(outputs_24hours, labels_24hours)
                        loss_resus = criterion_resus(outputs_resus, labels_resus)

                        # Reduce losses to scalar values
                        loss_30min = loss_30min.mean()
                        loss_24hours = loss_24hours.mean()
                        loss_resus = loss_resus.mean()

                       # print(f'loss_30min: {loss_30min.item()}, loss_24hours: {loss_24hours.item()}, loss_resus: {loss_resus.item()}')

                        loss = loss_30min + loss_24hours + loss_resus
                        loss.backward()

                        if sch == "ReduceLROnPlateau":
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                        loss_30min = loss_30min.mean()
                        loss_24hours = loss_24hours.mean()
                        loss_resus = loss_resus.mean()

                        train_loss_30min += loss_30min.item() * data.size(0)
                        train_loss_24hours += loss_24hours.item() * data.size(0)
                        train_loss_resus += loss_resus.item() * data.size(0)

                        # Calculate accuracy
                        _, predicted_30min = torch.max(outputs_30min.data, 1)
                        _, predicted_24hours = torch.max(outputs_24hours.data, 1)
                        _, predicted_resus = torch.max(outputs_resus.data, 1)

                        total_30min += labels_30min.size(0)
                        total_24hours += labels_24hours.size(0)
                        total_resus += labels_resus.size(0)


                        correct_30min += (predicted_30min == labels_30min).sum().item()
                        correct_24hours += (predicted_24hours == labels_24hours).sum().item()
                        correct_resus += (predicted_resus == labels_resus).sum().item()

                    train_loss_30min /= len(train_dataset)
                    train_loss_24hours /= len(train_dataset)
                    train_loss_resus /= len(train_dataset)

                    total_train_loss = train_loss_30min + train_loss_24hours + train_loss_resus

                    train_losses_30min.append(train_loss_30min)
                    train_losses_24hours.append(train_loss_24hours)
                    train_losses_resus.append(train_loss_resus)
                    train_losses.append(total_train_loss)


                    seconds = time.time() - st
                    minutes = seconds / 60

                    train_acc_30min = 100 * correct_30min / total_30min
                    train_acc_24hours = 100 * correct_24hours / total_24hours
                    train_acc_resus = 100 * correct_resus / total_resus

                    # Validation
                    model.eval()

                    val_correct_30min = 0
                    val_correct_24hours = 0
                    val_correct_resus = 0
                    val_total_30min = 0
                    val_total_24hours = 0
                    val_total_resus = 0

                    with torch.no_grad():
                        for data, labels in val_loader:
                            # data = data.unsqueeze(1)  # Add channel dimension
                            if torch.cuda.is_available():
                                # data, target = data.cuda(), labels.cuda()
                                data, labels = data.to(device), labels.to(device)

                            labels_30min, labels_24hours, labels_resus = labels[:, 0] , labels[:, 1] , labels[:, 2]
                            outputs_30min, outputs_24hours, outputs_resus = model(data)
                            
                            loss_30min = criterion_30min(outputs_30min, labels_30min)
                            loss_24hours = criterion_24hour(outputs_24hours, labels_24hours)
                            loss_resus = criterion_resus(outputs_resus, labels_resus)

                            loss_30min = loss_30min.mean()
                            loss_24hours = loss_24hours.mean()
                            loss_resus = loss_resus.mean()

                            val_loss_30min += loss_30min.item() * data.size(0)
                            val_loss_24hours += loss_24hours.item() * data.size(0)
                            val_loss_resus += loss_resus.item() * data.size(0)

                            # Calculate validation accuracy
                            _, predicted_30min = torch.max(outputs_30min.data, 1)
                            _, predicted_24hours = torch.max(outputs_24hours.data, 1)
                            _, predicted_resus = torch.max(outputs_resus.data, 1)

                            val_total_30min += labels_30min.size(0)
                            val_total_24hours += labels_24hours.size(0)
                            val_total_resus += labels_resus.size(0)

                            val_correct_30min += (predicted_30min == labels_30min).sum().item()
                            val_correct_24hours += (predicted_24hours == loss_24hours).sum().item()
                            val_correct_resus += (predicted_resus == loss_resus).sum().item()


                    val_loss_30min /= len(val_dataset)
                    val_loss_24hours /= len(val_dataset)
                    val_loss_resus /= len(val_dataset)

                    total_val_loss = val_loss_30min + val_loss_24hours + val_loss_resus

                    val_losses_30min.append(val_loss_30min)
                    val_losses_24hours.append(val_loss_24hours)
                    val_losses_resus.append(val_loss_resus)

                    val_losses.append(total_val_loss)

                    val_acc_30min = 100 * val_correct_30min / val_total_30min
                    val_acc_24hours = 100 * val_correct_24hours / val_total_24hours
                    val_acc_resus = 100 * val_correct_resus / val_total_resus


                    #print(f"Train Acc 30min: {train_acc_30min:.2f}%, Train Acc 24hours: {train_acc_24hours:.2f}%, \
                     #Val Acc 30min: {val_acc_30min:.2f}%, Val Acc 24hours: {val_acc_24hours:.2f}%")

                    print(f"TrainLoss-30min: {train_loss_30min:.2f}, TrainLoss-24hours: {train_loss_24hours:.2f},\
                    TrainLoss-Resus: {train_loss_resus:.2f},", f"\n ValLoss-30min: {val_loss_30min:.2f}, \
                     ValLoss-24hours: {val_loss_24hours:.2f},  ValLoss-Resus: {val_loss_resus:.2f}")

                    # Save the best model weights
                    if total_val_loss < best_val_loss:
                        best_val_loss = total_val_loss
                        best_model_weights = model.state_dict()
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    # Early stopping
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        break

                print(f"Total epochs: {epoch + 1}, Training time consumed: {minutes:.2f} minutes")

                # Plot the training and validation losses
                plt.figure(figsize=(20, 10))
                plt.plot(train_losses_30min, color='r', linewidth=2, label='Training Loss - 30 min')
                plt.plot(train_losses_24hours, color='r', linestyle='--', linewidth=2, label='Training Loss - 24h')
                plt.plot(train_losses_resus, color='r', linestyle='--', linewidth=2, label='Training Loss - Resus')

                plt.plot(val_losses_30min, color='b', linewidth=2, label='Validation Loss - 30m')
                plt.plot(val_losses_24hours, color='b', linestyle='--', linewidth=2, label='Validation Loss - 24h')
                plt.plot(val_losses_resus, color='b', linestyle='--', linewidth=2, label='Validation Loss - Resus')

                val_loss = [val_losses_30min[i] + val_losses_24hours[i] + val_losses_resus[i] for i in range(len(val_losses_30min))]
                plt.plot(val_loss, color='k', linestyle='-.', linewidth=3, label='Validation Loss - Total')

                train_loss = [train_losses_30min[i] + train_losses_24hours[i] + train_losses_resus[i] for i in range(len(train_losses_30min))]
                plt.plot(train_loss, color='g', linestyle='-.', linewidth=3, label='Training Loss - Total')

                plt.xlabel('Epoch', fontsize=16)
                plt.ylabel('Loss', fontsize=16)
                plt.title('Training and Validation Losses', fontsize=20)
                plt.legend(fontsize=16)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.savefig(f"{path}/LossCurve-TCN-3Classifiers.png")
                # plt.show()

                torch.save({'model': best_model_weights}, f"{path}/best_weights.dat")
                # Load the best model weights
                model.load_state_dict(best_model_weights)

                print("\nValidating the model.........")
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

                preds_30min, preds_24hours,  probs_30min, probs_24hours, label_30min, label_24hours = [], [], [], [], [], []
                preds_resus, probs_resus, label_resus = [], [], []

                with torch.no_grad():
                    for data, labels in val_loader:
                        if torch.cuda.is_available():
                            data, labels = data.to(device), labels.to(device)
                        labels_30min, labels_24hours, labels_resus = labels[:, 0] , labels[:, 1] , labels[:, 2]

                        outputs_30min, outputs_24hours, outputs_resus = model(data)

                        probabilities_outputs_30min = F.softmax(outputs_30min, dim=1)
                        probabilities_outputs_24hours = F.softmax(outputs_24hours, dim=1)
                        probabilities_outputs_resus = F.softmax(outputs_24hours, dim=1)

                        # Calculate accuracy
                        _, predicted_30min = torch.max(outputs_30min.data, 1)
                        _, predicted_24hours = torch.max(outputs_24hours.data, 1)
                        _, predicted_resus = torch.max(outputs_24hours.data, 1)

                        total_30min += labels_30min.size(0)
                        total_24hours += labels_24hours.size(0)
                        total_resus += labels_resus.size(0)


                        correct_30min += (predicted_30min == labels_30min).sum().item()
                        correct_24hours += (predicted_24hours == labels_24hours).sum().item()
                        correct_resus += (predicted_resus == labels_resus).sum().item()

                        label_30min.append(list(labels_30min.cpu().numpy()))
                        label_24hours.append(list(labels_24hours.cpu().numpy()))
                        label_resus.append(list(labels_resus.cpu().numpy()))

                        preds_30min.append(list(predicted_30min.cpu().numpy()))
                        preds_24hours.append(list(predicted_24hours.cpu().numpy()))
                        preds_resus.append(list(predicted_resus.cpu().numpy()))

                        probs_30min.append(list(np.around(probabilities_outputs_30min.detach().cpu().numpy(), decimals=4)))
                        probs_24hours.append(list(np.around(probabilities_outputs_24hours.detach().cpu().numpy(), decimals=4)))
                        probs_resus.append(list(np.around(probabilities_outputs_resus.detach().cpu().numpy(), decimals=4)))

                label_30min, preds_30min, probs_30min = convert_batch_list(label_30min), convert_batch_list(preds_30min), convert_batch_list(probs_30min)
                label_24hours, preds_24hours, probs_24hours = convert_batch_list(label_24hours), convert_batch_list(preds_24hours), convert_batch_list(probs_24hours)
                label_resus, preds_resus, probs_resus = convert_batch_list(label_resus), convert_batch_list(preds_resus), convert_batch_list(probs_resus)


                label_30min = [x+1 for x in label_30min]
                label_24hours = [x+1 for x in label_24hours]
                preds_30min = [x+1 for x in preds_30min]
                preds_24hours = [x+1 for x in preds_24hours]


                file_names = [im.split(".")[0] for im in val_loader.dataset.files]
                data = {"files": file_names, "ground_truth_30min": label_30min,  "predicted_30min": preds_30min, "prob_30min": probs_30min, \
                "ground_truth_24hours": label_24hours,"predicted_24hours": preds_24hours, "prob_24hours": probs_24hours, \
                "ground_truth_resus": label_resus,"predicted_resus": preds_resus, "prob_resus": probs_resus}

                dframe = pd.DataFrame(data)
                print("Length of validation data ", len(dframe))
                with pd.ExcelWriter(f"{path}/validation_3classes.xlsx") as wr:
                        dframe.to_excel(wr, index=False)

                test_acc_30min = 100 * correct_30min / total_30min
                test_acc_24hours = 100 * correct_24hours / total_24hours
                test_acc_resus = 100 * correct_resus / total_resus

                print(f'Valid-Accuracy 30min: {test_acc_30min:.2f}%, Valid-Accuracy 24hours: {test_acc_24hours:.2f}%, Valid-Accuracy Resus: {test_acc_resus:.2f}%, ')

                cm = confusion_matrix(label_30min, preds_30min)
                make_pretty_cm(cm, group_names=['1','2','3'], figsize=(8,8), title= f"{path}/Validation-30min-CM")

                cm = confusion_matrix(label_24hours, preds_24hours)
                make_pretty_cm(cm, group_names=['1','2','3'], figsize=(8,8), title= f"{path}/Validation-24hours-CM")

                cm = confusion_matrix(label_resus, preds_resus)
                make_pretty_cm(cm, group_names=['1','2'], figsize=(8,8), title= f"{path}/Validation-RESUS-CM")

                print("\nTesting the model.........")
                correct_30min = 0
                correct_24hours = 0
                correct_resus = 0

                total_30min = 0
                total_24hours = 0
                total_resus = 0

                test_acc_30min = 0
                test_acc_24hours = 0
                total_acc_resus = 0

                preds_30min, preds_24hours,  probs_30min, probs_24hours, label_30min, label_24hours = [], [], [], [], [], []
                preds_resus, probs_resus, label_resus = [], [], []

                with torch.no_grad():

                    for data, labels in test_loader:
                        if torch.cuda.is_available():
                            data, labels = data.to(device), labels.to(device)
                        labels_30min, labels_24hours, labels_resus = labels[:, 0] - 1, labels[:, 1] - 1, labels[:, 2]

                        outputs_30min, outputs_24hours, outputs_resus = model(data)

                        probabilities_outputs_30min = F.softmax(outputs_30min, dim=1)
                        probabilities_outputs_24hours = F.softmax(outputs_24hours, dim=1)
                        probabilities_outputs_resus = F.softmax(outputs_24hours, dim=1)

                        # Calculate accuracy
                        _, predicted_30min = torch.max(outputs_30min.data, 1)
                        _, predicted_24hours = torch.max(outputs_24hours.data, 1)
                        _, predicted_resus = torch.max(outputs_24hours.data, 1)

                        total_30min += labels_30min.size(0)
                        total_24hours += labels_24hours.size(0)
                        total_resus += labels_resus.size(0)

                        correct_30min += (predicted_30min == labels_30min).sum().item()
                        correct_24hours += (predicted_24hours == labels_24hours).sum().item()
                        correct_resus += (predicted_resus == labels_resus).sum().item()

                        label_30min.append(list(labels_30min.cpu().numpy()))
                        label_24hours.append(list(labels_24hours.cpu().numpy()))
                        label_resus.append(list(labels_resus.cpu().numpy()))

                        preds_30min.append(list(predicted_30min.cpu().numpy()))
                        preds_24hours.append(list(predicted_24hours.cpu().numpy()))
                        preds_resus.append(list(predicted_resus.cpu().numpy()))

                        probs_30min.append(list(np.around(probabilities_outputs_30min.detach().cpu().numpy(), decimals=4)))
                        probs_24hours.append(list(np.around(probabilities_outputs_24hours.detach().cpu().numpy(), decimals=4)))
                        probs_resus.append(list(np.around(probabilities_outputs_resus.detach().cpu().numpy(), decimals=4)))

                label_30min, preds_30min, probs_30min = convert_batch_list(label_30min), convert_batch_list(preds_30min), convert_batch_list(probs_30min)
                label_24hours, preds_24hours, probs_24hours = convert_batch_list(label_24hours), convert_batch_list(preds_24hours), convert_batch_list(probs_24hours)
                label_resus, preds_resus, probs_resus = convert_batch_list(label_resus), convert_batch_list(preds_resus), convert_batch_list(probs_resus)


                nlabel_30min = [x + 1 for x in label_30min]
                nlabel_24hours = [x + 1 for x in label_24hours]
                npreds_30min = [x + 1 for x in preds_30min]
                npreds_24hours = [x + 1 for x in preds_24hours]

                file_names = [im.split(".")[0] for im in test_loader.dataset.files]

                data = {"files": file_names, "ground_truth_30min": nlabel_30min,  "predicted_30min": npreds_30min, "prob_30min": probs_30min, \
                "ground_truth_24hours": nlabel_24hours,"predicted_24hours": npreds_24hours, "prob_24hours": probs_24hours, \
                "ground_truth_resus": label_resus,"predicted_resus": preds_resus, "prob_resus": probs_resus}

                dframe = pd.DataFrame(data)
                print("Length of test data ", len(dframe))
                with pd.ExcelWriter(f"{path}/test_3classes.xlsx") as wr:
                        dframe.to_excel(wr, index=False)

                test_acc_30min = 100 * correct_30min / total_30min
                test_acc_24hours = 100 * correct_24hours / total_24hours
                test_acc_resus = 100 * correct_resus / total_resus

                print(f'Test-Accuracy 30min: {test_acc_30min:.2f}%, Test-Accuracy 24hours: {test_acc_24hours:.2f}%, Test-Accuracy Resus: {test_acc_resus:.2f}%,')

                cm = confusion_matrix(label_30min, preds_30min)
                make_pretty_cm(cm, group_names=['1','2','3'], figsize=(12,12), title= f"{path}/Test-30min-CM")

                cm = confusion_matrix(label_24hours, preds_24hours)
                make_pretty_cm(cm, group_names=['1','2','3'], figsize=(12,12), title= f"{path}/Test-24hours-CM")

                cm = confusion_matrix(label_resus, preds_resus)
                make_pretty_cm(cm, group_names=['1','2'], figsize=(12,12), title= f"{path}/Test-RESUS-CM")

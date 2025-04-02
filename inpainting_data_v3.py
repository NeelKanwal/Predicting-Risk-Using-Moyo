# Author: Neel Kanwal, neel.kanwal0@gmail.com
# Inpanting attempt
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
        'size'   : 22}
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (24, 16)
matplotlib.use('Agg')

import seaborn as sns
sns.set_style("white")
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from torch.utils.data import DataLoader
import torch

from myfunctions import FHRDataset_v2_envelope, TransformerMaskedAutoencoder_inference, get_effective_signal_length, \
plot_interpolated, plot_interpolated_same, interpolate_missing_values_v4, interpolate_missing_values_v5,\
TransformerMaskedAutoencoder


# Set environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cuda_device = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

# Ensure CUDA is initialized
torch.cuda._lazy_init()

# Check if CUDA is available
if torch.cuda.is_available():
    # Set the device
    device = torch.cuda.current_device()
    print("Current CUDA device = ", device)
else:
    print("No CUDA GPUs are available")
    # Handle the case where no GPUs are available
    device = torch.device("cpu")
    print("Using CPU instead")
##

train_dir = "/nfs/...../training_with_envelope/"
val_dir = "/nfs/........./validation_with_envelope/"

val_dataset = FHRDataset_v2_envelope(val_dir, sequence_length=7200, normalization='minmax')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

def get_sample_at_index(dataloader, index):
    dataset = dataloader.dataset
    sample = dataset[index]
    return sample

def sample_to_dataframe(sample):
    combined_signal, fhr_original, interp_mask, outcomes = sample
    combined_signal = combined_signal.squeeze().numpy()
    fhr_original = fhr_original.squeeze().numpy()
    interp_mask = interp_mask.squeeze().numpy()
    
    # Create a DataFrame
    df = pd.DataFrame({
        'time_series': combined_signal[0],
        'fhr_original': fhr_original,
        'interp_mask': interp_mask
    })

    return df, outcomes


# Transformer with simple loss
model_sim = TransformerMaskedAutoencoder(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=60)
## New weights testing with 15%masking, input size=60 
best_model_wts = '/nfs/.........../10_10_2024_13_39_44/best_weights.dat'
model_sim.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])


# Transformer with simple loss, PREVOIUS
# model_sim = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=256, nhead=8, num_encoder_layers=4,num_decoder_layers=4, dim_feedforward=1024, patch_size=60)
# best_model_wts = '/nfs/s........../09_15_2024_12_35_08/best_weights.dat'
# model_sim.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])

# Transformer with hybrid loss
model_hybrid = TransformerMaskedAutoencoder(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=60)
## New weights testing with 10%masking 
best_model_wts = '/nfs/........../best_weights.dat'
model_hybrid.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])

# Transformer with hybrid loss, previous
# model_hybrid = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=256, nhead=8, num_encoder_layers=4,num_decoder_layers=4, dim_feedforward=1024, patch_size=60)
# best_model_wts = '/nfs/........../best_weights.dat'
# model_hybrid.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])


for index in [58,122,172]:
    print(f"Working for index: {index}")

    # Extract a sample to perform inpainting
    sample = get_sample_at_index(val_loader, index=index)
    df, outcomes = sample_to_dataframe(sample)
    outcome_30min, outcome_24hours, outcome_resus = outcomes.tolist()
    print("Sample loaded")

    # # Perform inpainting using both models
    # inpainted_signal_sim = inpaint_signal(df, model_sim, device)
    # inpainted_signal_hybrid = inpaint_signal(df, model_hybrid, device)

    # # Perform inpainting using both models
    inpainted_signal_sim, mse_sim = interpolate_missing_values_v5(df, model_sim, device)
    inpainted_signal_hybrid, mse_hybrid = interpolate_missing_values_v5(df, model_hybrid, device)


   

    # Convert all relevant data to numpy arrays
    fhr_original = df['fhr_original'].values
    time_series = df['time_series'].values
    interp_mask = df['interp_mask'].values.astype(bool)  # Convert to boolea
    masked_original_signal = time_series * (~interp_mask)
    length = get_effective_signal_length(time_series)

     # Create a new x-axis from -7200 to 0
    x_axis = np.arange(-len(df['fhr_original']), 0)[:length]


    #  # Create a SINGLE figure for all four signals
    # fig, ax = plt.subplots(1, 1, figsize=(32, 8), sharex=True)
    # ax.plot(x_axis, df['time_series'], color='green', linewidth=1.5, alpha=0.8, label='Linear Interpolation')
    # ax.plot(x_axis, masked_original_signal, color='blue', linewidth=2, label='Masked Version (Encoder)')
    # ax.plot(x_axis, inpainted_signal_sim, color='red', linewidth=1.5, alpha=0.8, label=f'Inpainted (Hybrid-30)- MSE: {mse_sim}')
    # ax.plot(x_axis, inpainted_signal_hybrid, color='orange', linewidth=1.5, label=f'Inpainted (Hybrid-60)- MSE: {mse_hybrid}')
    # ax.set_title("Original Signal vs. Interpolated and Inpainted Signals", fontsize=28)
    # ax.set_xlabel('Time to event (seconds)', fontsize=22)
    # ax.set_ylabel('Signal Amplitude', fontsize=22)
    # ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    # ax.grid(True, linestyle='--', alpha=0.7)
    # ax.set_xticks(np.arange(-7200, 1, 1200))
    # ax.set_xticklabels(np.arange(-7200, 1, 1200), fontsize=20)
    # ax.tick_params(axis='y', labelsize=20)
    # ax.set_xlim(-7200, 0)
    # ax.set_ylim(0, 1.1)

    # plt.tight_layout()
    # plt.savefig(f"3_inpainting_combined_{index}.png", dpi=300, bbox_inches='tight')
    # plt.close()

        # Figure 2: Separate subplots for each signal
    fig, axs = plt.subplots(5, 1, figsize=(30, 21), sharex=True)
    fig.suptitle(f"Original Signal vs Inpainted Signals", fontsize=28)
    # Time Series (Linear Interpolation)
    axs[0].plot(x_axis, time_series[:length], color='green', linewidth=1.5, alpha=0.8, label='Linear Interpolation')
    axs[0].set_ylabel('Signal Amplitude', fontsize=20)
    axs[0].grid(True, linestyle='--', alpha=0.8)
    axs[0].set_ylim(0, 1)
    axs[0].legend()
    axs[0].set_title('Linearly Interpolated Version')


     # Inpainted (Model Hybrid)
    axs[1].plot(x_axis, interp_mask[:length], 'black', label='Interpolation Mask')
    axs[1].set_ylabel('Signal Amplitude', fontsize=20)
    axs[1].grid(True, linestyle='--', alpha=0.8)
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].legend()
    axs[1].set_title('Intepolation locations')

    # FHR Original
    axs[2].plot(x_axis, masked_original_signal[:length], color='blue', linewidth=2, label='Masked Version (Encoder)')
    axs[2].set_ylabel('Signal Amplitude', fontsize=20)
    axs[2].grid(True, linestyle='--', alpha=0.8)
    axs[2].set_ylim(0, 1)
    axs[2].legend()
    axs[2].set_title('FHR Original Version')

     # Inpainted (Model Sim) # continous lines for values next to each other. 
    #plot_interpolated(axs[3], x_axis, df['time_series'], inpainted_signal_sim, interp_mask, 'red', 'Inpainted (Hybrid-30)')
    axs[3].plot(x_axis, inpainted_signal_sim, color='red', linewidth=1.5, alpha=0.8, label=f'Inpainted (Hybrid-60_15)') # reffered to as Model_2
    axs[3].set_ylabel('Signal Amplitude', fontsize=20)
    axs[3].grid(True, linestyle='--', alpha=0.8)
    axs[3].set_ylim(0, 1)
    axs[3].legend()
    axs[3].set_title(f'Inpainted using FHRTransformer (Hybrid-30) with MSE:{mse_sim:.5f}')

    # Inpainted (Model Hybrid)
    axs[4].plot(x_axis, inpainted_signal_hybrid, color='orange', linewidth=1.5, label=f'Inpainted (Hybrid-60_10)') # referred as model_1
    axs[4].set_ylabel('Signal Amplitude', fontsize=20)
    axs[4].grid(True, linestyle='--', alpha=0.8)
    axs[4].set_ylim(0, 1)
    axs[4].legend()
    axs[4].set_title(f'Inpainted using FHRTransformer (Hybrid-60) with MSE:{mse_hybrid:.5f}')

    # Set x-axis labels and ticks
    axs[4].set_xlabel('Time to event (seconds)', fontsize=20)
    plt.xticks(np.arange(-7200, 1, 1200), fontsize=16)
    plt.xlim(-7200, 0)


    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"3_Inpainting_comparison_subplots_{index}.png", dpi=300, bbox_inches='tight')
    plt.close()

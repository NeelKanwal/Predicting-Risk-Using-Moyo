# Author: Neel Kanwal, neel.kanwal0@gmail.com
# This script generates a new version of dataset using AI-inpainted method as described. 


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
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from myfunctions import FHRDataset_v2_envelope, TransformerMaskedAutoencoder_inference, get_effective_signal_length, \
plot_interpolated, plot_interpolated_same, interpolate_missing_values_v4, interpolate_missing_values_v5,\
TransformerMaskedAutoencoder
from scipy.signal import savgol_filter

# Transformer with simple loss, PREVOIUS
# model_sim = TransformerMaskedAutoencoder_inference(input_dim=1, d_model=256, nhead=8, num_encoder_layers=4,num_decoder_layers=4, dim_feedforward=1024, patch_size=60)

# model_sim.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])

###
#OLD HYBRID weights
#best_model_wts = '/nfs/........../09_15_2024_12_37_15/best_weights.dat'
# #New_weights, with patch size of 60
# best_model_wts = '/nfs/........../10_26_2024_04_25_07/best_weights.dat'
# #New_weights, with patch size of 30
# best_model_wts = '/nfs/s........../10_25_2024_12_47_52/best_weights.dat'
# NEW model with 5 layers and trained without bumps in learning curve. Patch INPUT = 60


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


# Perform recursive forecasting
start_index = 3600
end_index = 7200
step_size = 30
past_length = 3600
##

train_dir = "/nfs/...../labelled_data_with_envelope/training/"
val_dir = "/nfs/......./labelled_data_with_envelope/validation/"
test_dir = "/nfs/....../labelled_data_with_envelope/test/"

use_dir = test_dir

new_data_dir = "/nfs/......./labelled_data_with_envelope/interp_test/"
os.makedirs(new_data_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_dataset = FHRDataset_v2_envelope(use_dir, sequence_length=7200, normalization='minmax')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def get_sample_at_index(dataloader, index):
    dataset = dataloader.dataset
    sample = dataset[index]
    # Get the original file name 
    original_file_name = dataset.files[index]
    return sample, original_file_name

def sample_to_dataframe(sample):
    combined_signal, fhr_original, missing_mask, outcomes = sample
    combined_signal = combined_signal.squeeze().numpy()
    fhr_original = fhr_original.squeeze().numpy()
    interp_mask = missing_mask.squeeze().numpy()
    
    df = pd.DataFrame({
        'time_series': combined_signal[0],
        'fhr_original': fhr_original,
        'interp_mask': interp_mask})
    
    return df, outcomes

## NEW CODE HERE
model_sim = TransformerMaskedAutoencoder(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=60)
## New weights testing with 15%masking, input size=60 
best_model_wts = '/nfs/.........../10_10_2024_13_39_44/best_weights.dat'
model_sim.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])

# Transformer with hybrid loss
model_hybrid = TransformerMaskedAutoencoder(input_dim=1, d_model=512, nhead=16, num_encoder_layers=5,num_decoder_layers=5, dim_feedforward=1024, patch_size=60)
## New weights testing with 10%masking 
best_model_wts = '/nfs/................../best_weights.dat'
model_hybrid.load_state_dict(torch.load(best_model_wts, map_location=torch.device('cpu'))['model'])


## CREATING AN UPDATE VERSION TO CREATA A NEW DATASET WITH INTERPOLATTED VERSION.


for idx in tqdm(range(len(val_dataset)), desc="Processing files"):

    # Extract a sample to perform inpainting
    sample, original_file_name = get_sample_at_index(val_loader, index=idx)
    df, _  = sample_to_dataframe(sample)

    inpainted_signal_sim, _ = interpolate_missing_values_v5(df, model_sim, device)
    inpainted_signal_hybrid, _ = interpolate_missing_values_v5(df, model_hybrid, device)
    
    # Load the original npz file
    original_data = np.load(os.path.join(use_dir, original_file_name))
    new_data = dict(original_data)

    new_data['fhr_original_normalized'] = df['fhr_original']
    new_data['time_series_normalized'] = df['time_series']
    # new_data['interp_mask'] = df['interp_mask']
    new_data['inpainted_model_2'] = inpainted_signal_sim
    new_data['inpainted_model_1'] = inpainted_signal_hybrid
    
    # Save the new npz file in the new directory
    new_file_path = os.path.join(new_data_dir, original_file_name)
    np.savez(new_file_path, **new_data)

print(f"Interpolated dataset saved in {new_data_dir}")



    # # Create a new x-axis from -7200 to 0
    # x_axis = np.arange(-7200, 0)
    # # Plotting the fhr_original signal with line breaks for missing values
    # plt.figure(figsize=(18, 7))
    
    # # Plot real obtained values with line breaks for missing values (NaNs)
    # plt.plot(x_axis, df['fhr_original'], label='fhr_original', color='blue')
    # plt.xticks(np.arange(-7200, 1, 1200), fontsize=18)
    # plt.xlim(-7200, 0)
    # plt.tight_layout()
    # plt.xlabel('Time')
    # plt.ylabel('FHR Original')
    # plt.title('FHR Original Signal with Missing Values Indicated')
    # plt.legend()
    # plt.savefig(f"Interpolation_{index}.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # # Inpainted (Model Sim) # oldest with lines connecting interpolated values.
    # axs[2].plot(x_axis, df['fhr_original'], color='blue', linewidth=1, alpha=0.3)
    # axs[2].plot(x_axis[missing_mask], inpainted_signal_sim[missing_mask], color='red', linewidth=1)
    # axs[2].set_ylabel('Inpainted (Model Sim)', fontsize=14)
    # axs[2].grid(True, linestyle='--', alpha=0.7)

    # # Inpainted (Model Hybrid)
    # axs[3].plot(x_axis, df['fhr_original'], color='blue', linewidth=1, alpha=0.3)
    # axs[3].plot(x_axis[missing_mask], inpainted_signal_hybrid[missing_mask], color='orange', linewidth=1)
    # axs[3].set_ylabel('Inpainted (Model Hybrid)', fontsize=14)
    # axs[3].grid(True, linestyle='--', alpha=0.7)


     # # Inpainted (Model Sim) # older, shows separated dots for interpolated values.
    # axs[2].plot(x_axis, df['fhr_original'], color='blue', linewidth=1, alpha=0.5, label='FHR Original')
    # axs[2].scatter(x_axis[missing_mask], inpainted_signal_sim[missing_mask], color='red', s=10, label='Inpainted (Hybrid-30)')
    # axs[2].set_ylabel('Signal Amplitude', fontsize=14)
    # axs[2].grid(True, linestyle='--', alpha=0.8)
    # axs[2].legend()

    # # Inpainted (Model Hybrid)
    # axs[3].plot(x_axis, df['fhr_original'], color='blue', linewidth=1, alpha=0.5, label='FHR Original')
    # axs[3].scatter(x_axis[missing_mask], inpainted_signal_hybrid[missing_mask], color='orange', s=10, label='Inpainted (Hybrid-60)')
    # axs[3].set_ylabel('Signal Amplitude', fontsize=14)
    # axs[3].grid(True, linestyle='--', alpha=0.8)
    # axs[3].legend()




# def sample_to_dataframe_old(sample):
#     combined_signal, fhr_original, outcomes = sample
#     combined_signal = combined_signal.squeeze().numpy()
#     fhr_original = fhr_original.squeeze().numpy()
    
#     # Create missing mask where fhr_original is zero or NaN
#     missing_mask = (fhr_original == 0) | np.isnan(fhr_original)
    
#     df = pd.DataFrame({
#         'time_series': combined_signal[0],
#         'fhr_original': fhr_original,
#         'missing_mask': missing_mask
#     })
#     df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
    
#     # Replace missing values with NaN in fhr_original
#     df.loc[df['missing_mask'], 'fhr_original'] = np.nan
    
#     return df, outcomes

# def sample_to_dataframe(sample, lower_threshold=0.33, upper_threshold=0.88):
#     # upper threshold 0.88 ~ 176
#     # upper threshold 0.33 ~ 58
#     combined_signal, fhr_original, outcomes = sample
#     combined_signal = combined_signal.squeeze().numpy()
#     fhr_original = fhr_original.squeeze().numpy()
    
#     # Create missing mask where fhr_original is zero or NaN
#     missing_mask = (fhr_original == 0) | np.isnan(fhr_original)
    
#     # Clip noisy values in fhr_original using normalized range [0, 1]
#     fhr_original_clipped = np.clip(fhr_original, lower_threshold, upper_threshold)

#     df = pd.DataFrame({
#         'time_series': combined_signal[0],
#         'fhr_original': fhr_original_clipped,
#         'missing_mask': missing_mask})
    
#     # Replace missing values with NaN in fhr_original_clipped
#     df.loc[df['missing_mask'], 'fhr_original'] = np.nan
    
#     return df, outcomes
# Author: Neel Kanwal, neel.kanwal0@gmail.
# this file plots Original FHR signal, Cleaned version, envelopes from accelration signals and interpolation mask
# this was created to see some samples before actually doing inpaiting for interpolated version.

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from myfunctions import FHRDataset_v4_envelope


def explore_dataset(dataset, index, figsize=(30, 20)):
    # Get the specific file name
    file_name = dataset.files[index]
    
    # Get the specific item
    # combined_signal, fhr_original, interp_mask, _ = dataset[index]
    fhr_original, time_series, peak_envelope, savgol_envelope, interp_mask, signal_hybrid30, signal_hybrid60, _ = dataset[index]

    
    # Convert tensors to numpy arrays and remove batch dimension
    fhr_original = fhr_original.numpy()
    time_series = time_series.numpy()
    peak_envelope = peak_envelope.numpy()
    savgol_envelope = savgol_envelope.numpy()
    interp_mask = interp_mask.numpy()
    signal_hybrid30 = signal_hybrid30.numpy()
    signal_hybrid60 = signal_hybrid60.numpy()
  
    
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=figsize, sharex=True)
    
    # Calculate x-axis values
    x_axis = np.arange(-len(fhr_original) + 1, 1)  # +1 to include 0
    
    # FHR Original
    axes[0].plot(x_axis, fhr_original, 'b', label='FHR Original', linewidth=1.5)
    axes[0].set_title(f"File: {file_name}", fontsize=20)
    axes[0].set_ylabel("Normalized HR", fontsize=16)
    axes[0].legend(fontsize=20, loc='upper right')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
     # Time Series (Interpolated)
    axes[1].plot(x_axis, time_series, 'g', label='Interpolated and Denoised Version', linewidth=1.5)
    axes[1].set_ylabel("Normalized HR", fontsize=16)
    axes[1].legend(fontsize=20, loc='upper right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle='--', alpha=0.7)

       # Peak Envelope
    axes[2].plot(x_axis, signal_hybrid30, 'r', label='Inpainted (Hybrid-30)', linewidth=1.5)
    axes[2].set_ylabel("Normalized HR", fontsize=16)
    axes[2].legend(fontsize=20, loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Savgol Envelope
    axes[3].plot(x_axis, signal_hybrid60, 'm', label='Inpainted (Hybrid-60)', linewidth=1.5)
    axes[3].set_ylabel("Normalized HR", fontsize=16)
    axes[3].set_xlabel("Time to event (seconds)", fontsize=16)
    axes[3].legend(fontsize=20, loc='upper right')
    axes[3].grid(True, linestyle='--', alpha=0.7)

    # Interpolation Mask
    axes[4].plot(x_axis, interp_mask, 'k', label='Interp Mask', linewidth=1, drawstyle='steps-post')
    axes[4].set_ylabel("Interpolated (1) / Original (0)", fontsize=16)
    axes[4].legend(fontsize=20, loc='upper right')
    axes[4].set_yticks([0, 1])
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].grid(True, linestyle='--', alpha=0.7)
   
      # Peak Envelope
    axes[5].plot(x_axis, peak_envelope, 'r', label='Peak Envelope', linewidth=1.5)
    axes[5].set_ylabel("Values", fontsize=16)
    axes[5].legend(fontsize=20, loc='upper right')
    axes[5].grid(True, linestyle='--', alpha=0.7)
    
    # Savgol Envelope
    axes[6].plot(x_axis, savgol_envelope, 'm', label='Savgol Envelope', linewidth=1.5)
    axes[6].set_ylabel("Values", fontsize=16)
    axes[6].set_xlabel("Time to event (seconds)", fontsize=16)
    axes[6].legend(fontsize=20, loc='upper right')
    axes[6].grid(True, linestyle='--', alpha=0.7)

    # Set x-axis limits and ticks for all subplots
    for ax in axes:
        ax.set_xlim(-7200, 0)
        ax.set_xticks(np.arange(-7200, 1, 1800))  # Ticks every 30 minutes
    
    plt.tight_layout()
    plt.savefig(f"{file_name}_exploration.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Print some statistics
    print(f"\nFile: {file_name} statistics:")
    print(f"Total points: {len(time_series)}")
    print(f"Interpolated points: {interp_mask.sum()}")
    print(f"Percentage interpolated: {interp_mask.mean()*100:.2f}%")

# Usage
val_dir = "/nfs/student/neel/MoYo_processed_data/validation_interpolated_with_envelope/"
dataset = FHRDataset_v4_envelope(val_dir)

# Specify the index of the file you want to explore
file_index = 8  # Change this to explore different files
explore_dataset(dataset, file_index)
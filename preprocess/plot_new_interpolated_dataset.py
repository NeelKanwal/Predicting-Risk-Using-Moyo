# Author: Neel Kanwal, neel.kanwal0@gmail.com
# Creates plots from new version of dataset with AI-interpolated signals. 


import numpy as np
import matplotlib.pyplot as plt
import os

# Set the path to the new directory containing the interpolated npz files
new_data_dir = "/nfs/student/neel/MoYo_processed_data/labelled_data_with_envelope/interp_test/"

def load_and_plot_file(file_path):
    # Load the npz file
    data = np.load(file_path)
    
    # Extract the required signals
    fhr_true = data['fhr_original_normalized']
    fhr_original = data['time_series_normalized']  # Note the typo in 'orignial'
    mask = data['interp_mask']  # Note the typo in 'orignial'
    inpainted_hybrid_30 = data['inpainted_model_1']
    inpainted_hybrid_60 = data['inpainted_model_2']
    
    # Function to pad array to 7200 samples
    def pad_to_7200(arr):
        if len(arr) < 7200:
            return np.pad(arr, (0, 7200 - len(arr)), mode='constant', constant_values=np.nan)
        else:
            return arr[:7200]
    
    # Pad all signals to 7200 samples
    fhr = pad_to_7200(fhr_true)
    mask = pad_to_7200(mask)
    fhr_original = pad_to_7200(fhr_original)
    inpainted_hybrid_30 = pad_to_7200(inpainted_hybrid_30)
    inpainted_hybrid_60 = pad_to_7200(inpainted_hybrid_60)
    
    # Create x-axis
    x_axis = np.arange(-len(fhr_original), 0)
    
    # Create the plot with three subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f"Signal Comparison - {os.path.basename(file_path)}", fontsize=16)
    
    # Plot FHR Original

    ax1.plot(x_axis, fhr, color='blue', label='FHR Original')
    ax1.plot(x_axis, mask, color='black', alpha=0.6, label ="Interpolation Mask")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_ylabel('FHR Original')
    ax1.grid(True)
    ax1.legend()
    

    ax2.plot(x_axis, fhr_original, color='blue', label='FHR Original (Cleaned and Normalized)')
    ax2.set_ylim(0, 1)
    # ax1.set_xlim(-7200, 0)
    ax2.set_ylabel('FHR Original')
    ax2.grid(True)
    ax2.legend()
    
    # Plot inpainted_hybrid_30
    # ax2.plot(x_axis, fhr_original, color='blue', alpha=0.5, label='FHR Original')
    ax3.plot(x_axis, inpainted_hybrid_30, color='red', label='Inpainted (Model 1)')
    ax3.set_ylabel('Inpainted (Hybrid-30)')
    ax3.set_ylim(0, 1)
    # ax2.set_xlim(-7200, 0)
    ax3.grid(True)
    ax3.legend()
    
    # Plot inpainted_hybrid_60
    # ax3.plot(x_axis, fhr_original, color='blue', alpha=0.5, label='FHR Original')
    ax4.plot(x_axis, inpainted_hybrid_60, color='green', label='Inpainted (Model 2)')
    ax4.set_ylabel('Inpainted (Hybrid-60)')
    ax4.grid(True)
    ax4.set_ylim(0, 1)
    # ax3.set_xlim(-7200, 0)
    ax4.legend()
    
    # Set x-axis label
    ax4.set_xlabel('Time to event (seconds)')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"signal_comparison_{os.path.basename(file_path)}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Get list of all npz files in the new directory
npz_files = [f for f in os.listdir(new_data_dir) if f.endswith('.npz')]
# Plot the first 5 files (or fewer if there are less than 5)
number_of_files = 5
for file_name in npz_files[:number_of_files]:
    file_path = os.path.join(new_data_dir, file_name)
    load_and_plot_file(file_path)

print(f"Plotted {min(number_of_files, len(npz_files))} files from {new_data_dir}")
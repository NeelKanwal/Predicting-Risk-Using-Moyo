# Author: Neel Kanwal, neel.kanwal0@gmail.
# Check percantage of interpolation

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random

train_data_dir = "/nfs/...../training_with_envelope/"
val_data_dir = "/nfs/....../validation_with_envelope/"  
# Add your validation directory path here

# def import_mat(filename):
#     try:
#         imported_mat = sio.loadmat(filename)
#         cleaninfo_cleanFHR = imported_mat['moyoData']['cleanInfo'][0][0][0]['cleanFHR'][0]
#         cleaninfo_interMask = imported_mat['moyoData']['cleanInfo'][0][0][0]['masks'][0][0][0]['interpolated']
#         return cleaninfo_cleanFHR, cleaninfo_interMask
#     except Exception as e:
#         print(f"Error reading file {filename}: {str(e)}")
#         return None, None

def import_npz(filename):
    try:
        npz_data = np.load(filename)
        time_series = npz_data['time_series']
        interp_mask = npz_data['interp_mask']
        return time_series, interp_mask
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None, None

def calculate_interpolation_percentage(cleanFHR, interMask):
    total_samples = len(cleanFHR)
    interpolated_samples = np.sum(interMask)
    return (interpolated_samples / total_samples) * 100

def process_directory(data_dir):
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    print(f"Total files in {data_dir}: {len(npz_files)}")
    
    full_percentages = []
    last_hour_percentages = []

    for filename in npz_files:
        time_series, interp_mask = import_npz(os.path.join(data_dir, filename))
        if time_series is None or interp_mask is None:
            continue
        
        full_percentage = calculate_interpolation_percentage(time_series, interp_mask)
        full_percentages.append(full_percentage)
        
        last_hour_time_series = time_series[-7200:]
        last_hour_interp_mask = interp_mask[-7200:]
        last_hour_percentage = calculate_interpolation_percentage(last_hour_time_series, last_hour_interp_mask)
        last_hour_percentages.append(last_hour_percentage)
    
    return full_percentages, last_hour_percentages

def create_histogram(full_percentages, last_hour_percentages, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(full_percentages, bins=20, edgecolor='black')
    ax1.set_title(f'{title} - Interpolation Percentage (Full Signal)')
    ax1.set_xlabel('Percentage of Interpolated Data')
    ax1.set_ylabel('Number of Files')

    ax2.hist(last_hour_percentages, bins=20, edgecolor='black')
    ax2.set_title(f'{title} - Interpolation Percentage (Last Hour)')
    ax2.set_xlabel('Percentage of Interpolated Data')
    ax2.set_ylabel('Number of Files')

    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_interpolation_percentage_histograms.png")
    plt.close()

    print(f"Histogram plots saved as '{title.lower().replace(' ', '_')}_interpolation_percentage_histograms.png'")

# Process both directories
train_full, train_last_hour = process_directory(train_data_dir)
val_full, val_last_hour = process_directory(val_data_dir)

# Combine results
combined_full = train_full + val_full
combined_last_hour = train_last_hour + val_last_hour

# Print statistics
for dataset, full, last_hour in [("Training", train_full, train_last_hour), 
                                 ("Validation", val_full, val_last_hour), 
                                 ("Combined", combined_full, combined_last_hour)]:
    print(f"\n{dataset} dataset statistics:")
    print(f"Total files processed: {len(full)}")
    print(f"Average interpolation percentage (full signal): {np.mean(full):.2f}%")
    print(f"Average interpolation percentage (last hour): {np.mean(last_hour):.2f}%")

# Create histogram plots
create_histogram(train_full, train_last_hour, "Training Data")
create_histogram(val_full, val_last_hour, "Validation Data")
create_histogram(combined_full, combined_last_hour, "Combined Dataset")


def get_samples_with_high_interpolation(data_dir, threshold=80):
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    high_interpolation_samples = []

    for filename in npz_files:
        time_series, interp_mask = import_npz(os.path.join(data_dir, filename))
        if time_series is None or interp_mask is None:
            continue
        
        percentage = calculate_interpolation_percentage(time_series, interp_mask)
        if percentage > threshold:
            high_interpolation_samples.append((filename, percentage))

    return high_interpolation_samples

def plot_high_interpolation_sample(data_dir, sample_filename):
    time_series, interp_mask = import_npz(os.path.join(data_dir, sample_filename))
    if time_series is None or interp_mask is None:
        return

    # Plot the sample
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8), sharex=True)

    x_axis = np.arange(len(time_series))

    axes[0].plot(x_axis, time_series, 'b', label='Time Series', linewidth=1.5)
    axes[0].set_title(f"File: {sample_filename}", fontsize=20)
    axes[0].set_ylabel("Value", fontsize=16)
    axes[0].legend(fontsize=20, loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(x_axis, interp_mask, 'g', label='Interpolation Mask', linewidth=1.5)
    axes[1].set_ylabel("Interpolation Mask", fontsize=16)
    axes[1].legend(fontsize=20, loc='upper right')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Set x-axis limits and ticks for all subplots
    for ax in axes:
        ax.set_xlim(len(time_series) - 1, -1, -1)  # Reverse x-axis
        tick_interval = max(1, len(time_series) // 4)  # Adjust tick interval based on data length
        ticks = np.arange(len(time_series) - 1, -1, -tick_interval)  # Generate ticks in reverse order
        ax.set_xticks(ticks)
        ax.set_xticklabels(-ticks, rotation=45)  # Label ticks with negative values

    plt.tight_layout()
    plt.savefig(f"{sample_filename}_high_interpolation_plot.png")
    plt.close()

    print(f"Plot saved as '{sample_filename}_high_interpolation_plot.png'")

# Get samples with high interpolation
train_high_interpolation_samples = get_samples_with_high_interpolation(train_data_dir)
val_high_interpolation_samples = get_samples_with_high_interpolation(val_data_dir)

# Combine results
combined_high_interpolation_samples = train_high_interpolation_samples + val_high_interpolation_samples

# Randomly select a sample
if combined_high_interpolation_samples:
    selected_sample = random.choice(combined_high_interpolation_samples)
    plot_high_interpolation_sample(train_data_dir if selected_sample[0] in os.listdir(train_data_dir) else val_data_dir, selected_sample[0])
else:
    print("No samples with more than 80% interpolated values found.")


# Randomly select a sample
if combined_high_interpolation_samples:
    selected_sample = random.choice(combined_high_interpolation_samples)
    plot_high_interpolation_sample(train_data_dir if selected_sample[0] in os.listdir(train_data_dir) else val_data_dir, selected_sample[0])
else:
    print("No samples with more than 80% interpolated values found.")
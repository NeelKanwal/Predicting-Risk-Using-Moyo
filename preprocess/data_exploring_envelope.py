# Author: Neel Kanwal, neel.kanwal0@gmail.

import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import os
# from myfunctions import import_mat
import scipy.io as sio
from scipy.signal import hilbert, find_peaks, savgol_filter
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.5

num_of_files = 1  # You can change this to any number
data_dir = "/nfs/student/neel/MoYo_processed_data/total_data_mat/"

mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
selected_files = np.random.choice(mat_files, size=num_of_files, replace=False)

outcome_dic = {1: 'Normal',
               2: 'NICU',
               3: 'Dead',
               4: 'Dead (FSB)'}

def import_mat(filename):
    # Import data from *.mat file

    imported_mat = sio.loadmat(filename)
    acc_enrg = imported_mat['moyoData']['acc_enrg'][0][0]
    acc_x_mg = imported_mat['moyoData']['acc_x_mg'][0][0]
    acc_y_mg = imported_mat['moyoData']['acc_y_mg'][0][0]
    acc_z_mg = imported_mat['moyoData']['acc_z_mg'][0][0]
    fhr_original = imported_mat['moyoData']['fhr'][0][0]
    fhr_qual = imported_mat['moyoData']['fhr_qual'][0][0]
    outcome = imported_mat['moyoData']['outcome'][0][0][0][0]

    cleaninfo_cleanFHR = imported_mat['moyoData']['cleanInfo'][0][0][0]['cleanFHR'][0]
    cleaninfo_missingMask = imported_mat['moyoData']['cleanInfo'][0][0][0]['masks'][0][0][0]['missingSamples']
    cleaninfo_interMask = imported_mat['moyoData']['cleanInfo'][0][0][0]['masks'][0][0][0]['interpolated']
    
    return acc_enrg, acc_x_mg, acc_y_mg, acc_z_mg, cleaninfo_cleanFHR, fhr_original, cleaninfo_missingMask, cleaninfo_interMask, outcome

def smooth_signal(signal, window_size=101, poly_order=3):
    return savgol_filter(signal.flatten(), window_size, poly_order)
def smooth_envelope(signal, window_size=None, poly_order=3):
    signal = signal.flatten()
    if window_size is None:
        window_size = len(signal) // 100
    window_size = max(poly_order + 2, min(window_size, len(signal) - 1))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < poly_order + 2:
        poly_order = window_size - 2
    return savgol_filter(signal, window_size, poly_order)

def rms_envelope(x, window_size):
    x = x.flatten()
    window_size = min(window_size, len(x))
    return np.sqrt(np.convolve(x**2, np.ones(window_size)/window_size, mode='same'))

def peak_envelope(x, smoothing_window=101):
    x = x.flatten()
    peaks, _ = find_peaks(x)
    if len(peaks) < 2:
        return np.zeros_like(x)
    envelope = np.interp(np.arange(len(x)), peaks, x[peaks])
    return smooth_signal(envelope, smoothing_window)

def downsample(signal, target_length):
    original_indices = np.arange(len(signal))
    downsampled_indices = np.linspace(0, len(signal) - 1, target_length)
    interpolator = interp1d(original_indices, signal, kind='linear')
    return interpolator(downsampled_indices)

selected_files = ["M16_H09601_0.mat"] # M16_T03526_1.mat, M16_H09601_0.mat
for filename in selected_files:
    acc_enrg, acc_x_mg, acc_y_mg, acc_z_mg, cleaninfo_cleanFHR, fhr_original, cleaninfo_missingMask, cleaninfo_interMask, outcome  = \
        import_mat(os.path.join(data_dir, filename))

    # Calculate acceleration magnitude
    acc_squared_sum = np.square(acc_x_mg) + np.square(acc_y_mg) + np.square(acc_z_mg)
    acc_magnitude = np.sqrt(np.maximum(acc_squared_sum, 0)).flatten()
    acc_magnitude_smooth = smooth_signal(acc_magnitude, window_size=1001, poly_order=3)

    print(f"File: {filename}")
    print(f"FHR shape: {fhr_original.shape}")
    print(f"Acceleration magnitude shape: {acc_magnitude.shape}")

    # Create envelopes
    smooth_env =smooth_signal(acc_magnitude_smooth, window_size=2001, poly_order=3)
    peak_env = peak_envelope(acc_magnitude_smooth, smoothing_window=1001)
    window_size_rms = len(acc_magnitude) // 10  # Use 10% of signal length for RMS window
    rms_env = rms_envelope(acc_magnitude_smooth, window_size_rms)
    rms_env = smooth_signal(rms_env, window_size=1001, poly_order=3)  # Additional smoothing

    # Downsample envelopes to match FHR length
    target_length = len(cleaninfo_cleanFHR)
    acc_magnitude_downsampled = downsample(acc_magnitude, target_length)
    smooth_env_downsampled = downsample(smooth_env, target_length)
    peak_env_downsampled = downsample(peak_env, target_length)
    rms_env_downsampled = downsample(rms_env, target_length)

    # Create a new figure for each file
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(50, 30), sharex=True)

    # First subplot: FHR and Clean FHR
    ax1 = axes[0]
    ax1.plot(fhr_original, 'b', label='FHR Original', linewidth=2)
    ax1.set_title(f"File: {filename.split('.')[0]} and Outcome = {outcome} with {outcome_dic[outcome]}", fontsize=28)
    ax1.set_ylabel("HR (bpm)", fontsize=28)
    ax1.legend(fontsize=32, loc='upper right')
    ax1.tick_params(axis='both', which='major', labelsize=28)

    # Second subplot: Savitzky-Golay filter envelope
    ax2 = axes[1]
    ax2.plot(cleaninfo_cleanFHR, 'g', label='FHR Cleaned', linewidth=2)
    ax2.set_ylabel("HR (bpm)", fontsize=28)
    ax2.legend(fontsize=32, loc='upper right')
    ax2.tick_params(axis='both', which='major', labelsize=28)

    # Third subplot: Peak envelope
    ax3 = axes[2]
    ax3.plot(cleaninfo_missingMask, 'k', label='Mising Mask', linewidth=1, alpha=0.9)
    ax3.set_ylabel("Bool", fontsize=28)
    ax3.legend(fontsize=32, loc='upper right')
    ax3.tick_params(axis='both', which='major', labelsize=28)

    # Fourth subplot: RMS envelope
    ax4 = axes[3]
    ax4.plot(cleaninfo_interMask, 'k', label='Interpolation Mask', linewidth=1, alpha=0.9)
    ax4.set_xlabel("Time (s)", fontsize=28)
    ax4.set_ylabel("Bool", fontsize=28)
    ax4.legend(fontsize=32, loc='upper right')
    ax4.tick_params(axis='both', which='major', labelsize=28)



    # # First subplot: FHR and Clean FHR
    # ax1 = axes[0]
    # ax1.plot(fhr, 'b', label='FHR', linewidth=2, alpha=0.7)
    # ax1.plot(cleaninfo_cleanFHR, 'g', label='Clean FHR', linewidth=3)
    # ax1.set_title(f"File: {filename.split('.')[0]} and Outcome = {outcome} with {outcome_dic[outcome]}", fontsize=28)
    # ax1.set_ylabel("HR (bpm)", fontsize=28)
    # ax1.legend(fontsize=32, loc='upper right')
    # ax1.tick_params(axis='both', which='major', labelsize=28)

    # # Second subplot: Savitzky-Golay filter envelope
    # ax2 = axes[1]
    # ax2.plot(acc_magnitude_downsampled, 'k', label='Acceleration Magnitude', linewidth=1, alpha=0.3)
    # ax2.plot(smooth_env_downsampled, 'r', label='Savgol Envelope', linewidth=3)
    # ax2.set_ylabel("Acceleration (mg)", fontsize=28)
    # ax2.legend(fontsize=32, loc='upper right')
    # ax2.tick_params(axis='both', which='major', labelsize=28)

    # # Third subplot: Peak envelope
    # ax3 = axes[2]
    # ax3.plot(acc_magnitude_downsampled, 'k', label='Acceleration Magnitude', linewidth=1, alpha=0.3)
    # ax3.plot(peak_env_downsampled, 'g', label='Peak Envelope', linewidth=3)
    # ax3.set_ylabel("Acceleration (mg)", fontsize=28)
    # ax3.legend(fontsize=32, loc='upper right')
    # ax3.tick_params(axis='both', which='major', labelsize=28)

    # # Fourth subplot: RMS envelope
    # ax4 = axes[3]
    # ax4.plot(acc_magnitude_downsampled, 'k', label='Acceleration Magnitude', linewidth=1, alpha=0.3)
    # ax4.plot(rms_env_downsampled, 'm', label=f'RMS Envelope (original window={window_size_rms})', linewidth=3)
    # ax4.set_xlabel("Time (s)", fontsize=28)
    # ax4.set_ylabel("Acceleration (mg)", fontsize=28)
    # ax4.legend(fontsize=32, loc='upper right')
    # ax4.tick_params(axis='both', which='major', labelsize=28)


    plt.tight_layout()
    plt.savefig(f"MaskPlotting_{filename.split('.')[0]}.png", dpi=300, bbox_inches='tight')
    # plt.close()

    print(f"Processed file: {filename}")
    print("----------------------------")
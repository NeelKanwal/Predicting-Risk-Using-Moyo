# Author: Neel Kanwal, neel.kanwal0@gmail.com
# Function to load data from mat files and generate plots for visualization

import matplotlib.pyplot as plt
import numpy as np
import os
from myfunctions import import_mat
import scipy.io as sio
from scipy.signal import hilbert

def import_mat(filename):
    # Import data from *.mat file
    imported_mat = sio.loadmat(filename)
    acc_enrg = imported_mat['moyoData']['acc_enrg'][0][0]
    acc_x_mg = imported_mat['moyoData']['acc_x_mg'][0][0]
    acc_y_mg = imported_mat['moyoData']['acc_y_mg'][0][0]
    acc_z_mg = imported_mat['moyoData']['acc_z_mg'][0][0]
    fhr = imported_mat['moyoData']['fhr'][0][0]
    fhr_qual = imported_mat['moyoData']['fhr_qual'][0][0]
    outcome = imported_mat['moyoData']['outcome'][0][0][0][0]
    cleaninfo_cleanFHR = imported_mat['moyoData']['cleanInfo'][0][0][0]['cleanFHR'][0]
    cleaninfo_interpolated = imported_mat['moyoData']['cleanInfo'][0][0][0]['masks'][0][0][0]['interpolated']
    
    return acc_enrg, acc_x_mg, acc_y_mg, acc_z_mg, fhr, fhr_qual, outcome, cleaninfo_cleanFHR, cleaninfo_interpolated

num_of_files = 1  # You can change this to any number
data_dir = "D:\\Moyo\\fhr_data\\old_data"

mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
selected_files = np.random.choice(mat_files, size=num_of_files, replace=False)

outcome_dic = {1: 'Normal',
               2:  'NICU',
               3: 'Dead',
               4: 'Dead (FSB)' }

for filename in selected_files:
    acc_enrg, acc_x_mg, acc_y_mg, acc_z_mg, fhr, fhr_qual, outcome, cleaninfo_cleanFHR, cleaninfo_interpolated = \
    import_mat(os.path.join(data_dir, filename))
    
    # Create a new figure for each file
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(60, 30))  # Increased width for better visibility
    
    # First subplot: FHR, CleanFHR, and Interpolated FHR
    ax1 = axes[0]
    ax1.plot(fhr, 'b', label='FHR', linewidth=2)
    ax1.plot(cleaninfo_cleanFHR, 'g', label='Clean FHR', linewidth=3)
    # ax1.plot(cleaninfo_interpolated, 'r', label='Interp. FHR', linewidth=1)
    ax1.set_title(f"File: {filename.split('.')[0]} and Outcome = {outcome} with {outcome_dic[outcome]}", fontsize=36)
    ax1.set_ylabel("HR (bpm)", fontsize=28)
    ax1.legend(fontsize=28)
    ax1.tick_params(axis='both', which='major', labelsize=28)

    # Second subplot: FHR Quality
    ax2 = axes[1]
    ax2.plot(fhr_qual, 'k', label='FHR Quality', linewidth=1)
    ax2.set_ylabel("Quality", fontsize=28)
    ax2.legend(fontsize=28)
    ax2.tick_params(axis='both', which='major', labelsize=28)

    # Third subplot: Acceleration values and acc_enrg
    # Calculate the new acceleration energy
    acc_squared_sum = np.square(acc_x_mg) + np.square(acc_y_mg) + np.square(acc_z_mg)
    new_acc_enrg = np.sqrt(np.maximum(acc_squared_sum, 0))
    # Calculate the energy envelope using Hilbert transform
    energy_envelope = np.abs(hilbert(new_acc_enrg))

    ax3 = axes[2]
    ax3.plot(acc_x_mg, 'r', label='Acc X', linewidth=1)
    ax3.plot(acc_y_mg, 'g', label='Acc Y', linewidth=1)
    ax3.plot(acc_z_mg, 'b', label='Acc Z', linewidth=1)
    ax3.plot(acc_enrg, 'k', label='Acc Energy (from mat)', linewidth=3)
    # ax3.plot(new_acc_enrg, 'm', label='Acc Energy (Calculated)', linewidth=4)
    ax3.set_xlabel("Time (s)", fontsize=28)
    ax3.set_ylabel("Acceleration (mg)", fontsize=28)
    ax3.legend(fontsize=24)
    ax3.tick_params(axis='both', which='major', labelsize=28)

    ax4 = axes[3]
    ax4.plot(acc_enrg, 'k', label='Acc Energy (from mat)', linewidth=3)
    ax4.plot(new_acc_enrg, 'm', label='Acc Energy (Calculated)', linewidth=4)
    ax4.plot(energy_envelope, 'c', label='Energy Envelope', linewidth=2, alpha=0.5)
    ax4.set_xlabel("Time (s)", fontsize=28)
    ax4.set_ylabel("Acceleration (mg)", fontsize=28)
    ax4.legend(fontsize=24)
    ax4.tick_params(axis='both', which='major', labelsize=28)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"fhr_analysis_{filename.split('.')[0]}.png", dpi=300, bbox_inches='tight')
    
    # Show the plot (optional, comment out if you don't want to display it)
    plt.show()
    
    # Close the figure to free up memory
    plt.close(fig)
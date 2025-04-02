## Author: Neel Kanwal, neel.kanwal0@gmail.com
# This was used to divide dataset and extract information from excel sheet to add in the npz files


import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder
np.seterr(divide='raise', invalid='raise')  # Raise exceptions instead of warnings


excel_path = "D:\\Moyo\\matched_details-modified.xlsx"
data_dir = "D:\\Moyo\\fhr_data\\old_data"
sav_dir = "D:\\Moyo\\fhr_data\\old_data\\processed"

# Create label encoders
# Load the Excel sheet
df = pd.read_excel(excel_path)

# label_encoders = {}
# for col in ['Apgar_5min', 'Resuscitation', 'Stimulation', 'Suction', 'BMV', 'Outcome_30min', 'Outcome_24hours']:
#     label_encoders[col] = LabelEncoder()
#     df[col] = label_encoders[col].fit_transform(df[col])


# Iterate over file names
count = 0
no_clean_count = 0

file_names = [a for a in os.listdir(data_dir) if a.endswith(".mat")]
for file_name in file_names:
    try:
        mat_file = scipy.io.loadmat(os.path.join(data_dir, file_name))
        time_series = mat_file['moyoData']['cleanInfo'][0][0][0]['cleanFHR'][0]

        fname = file_name.strip(".mat")

        labels = df.loc[df['filename'] == fname, ['Apgar_5min', 'Resuscitation', 'Stimulation', 'Suction', 'BMV', 'nOutcome_30min',
                                          'nOutcome_24hours']].values[0]

        # Save data as NPZ file
        np.savez_compressed(os.path.join(sav_dir, fname + '.npz'), time_series=time_series, labels=labels)
        count += 1

    except:
        print(f"CleanFHR for {file_name} does not exist.")
        no_clean_count += 1


print("Created npz files for, ", count)
# print("No CleanFHR available for, ", no_clean_count)
## SPLIT DATASET AFTER NPZ FILES ARE CREATED USING THE FOLLOWING.
#
#
import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# # Directory containing the NPZ files
data_dir = "D:\\...\\processed"
train_dir = "D:\\....\\training"
val_dir = "D:\\....\\validation"
test_dir = "D:\\....\\test"
#

labels = []
filenames = []

# filenames_3 = []
# labels_3 = []

files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
# Load data and labels from NPZ

for file in files:
    if file.endswith('.npz'):
        file_path = os.path.join(data_dir, file)
        npz_data = np.load(file_path)
        label = npz_data['labels'][-2]  # Assuming 'Outcome_24hours' is the last label
        # we takt outcome 30 mins
        # sequence of labels [''Apgar_5min', 'Resuscitation', Stimulation', 'Suction', 'BMV', 'Outcome_30min', 'Outcome_24hours']
        labels.append(label)
        filenames.append(file)


# Convert data and labels to numpy arrays
labels = np.array(labels)
# Count the occurrences of each class
unique_labels, counts = np.unique(labels, return_counts=True)


train_filenames, temp_filenames, train_labels, temp_labels = train_test_split(filenames, labels, test_size=0.2, stratify=labels, random_state=2)
val_filenames, test_filenames, val_labels, test_labels = train_test_split(temp_filenames, temp_labels, test_size=0.5, stratify=temp_labels, random_state=2)


# train_filenames.extend(filenames_3[:3])
# val_filenames.extend(filenames_3[3:5])
# test_filenames.extend(filenames_3[5:])
#
# train_labels = np.append(train_labels, labels_3[:3])
# val_labels = np.append(val_labels, labels_3[3:5])
# test_labels = np.append(test_labels, labels_3[5:])

# Print the sizes of the subsets
print(f"Training set size: {len(train_filenames)}")
print(f"Validation set size: {len(val_filenames)}")
print(f"Test set size: {len(test_filenames)}")


# Print the distribution of 'Outcome_30min' in each subset
print("\nTraining set distribution:")
print(np.unique(train_labels, return_counts=True))

print("\nValidation set distribution:")
print(np.unique(val_labels, return_counts=True))

print("\nTest set distribution:")
print(np.unique(test_labels, return_counts=True))


# Copy files to respective directories
for file in train_filenames:
    src_path = os.path.join(data_dir, file)
    dst_path = os.path.join(train_dir, file)
    shutil.copy(src_path, dst_path)

for file in val_filenames:
    src_path = os.path.join(data_dir, file)
    dst_path = os.path.join(val_dir, file)
    shutil.copy(src_path, dst_path)

for file in test_filenames:
    src_path = os.path.join(data_dir, file)
    dst_path = os.path.join(test_dir, file)
    shutil.copy(src_path, dst_path)

#
# ###########################################################
# ###########################################################
#
# unique_labels
# array([1, 2, 3], dtype=int64)
# counts
# array([3418,  257,   19], dtype=int64)

# # outcome_counts = {'training': {1: 2734, 2: 129, 3: 3, 4: 86},
# #                   'validation': {1: 342, 2: 16, 3: 2, 4: 11},
# #                   'test': {1: 342, 2: 16, 3: 2, 4: 11}}

# NEW
outcome_counts = {'training': {1: 2734, 2: 206, 3: 15},
                  'validation': {1: 342, 2: 25, 3: 2},
                  'test': {1: 342, 2: 26, 3: 2,}}
# #
# # folders = list(outcome_counts.keys())
# # outcomes = list(outcome_counts['training'].keys())
# #
# # fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
# # x = np.arange(len(folders))  # the label locations
# # width = 0.15  # the width of the bars
# # multiplier = 0
# #
# # hatch_patterns = ['/', '\\', 'x', '+']
# #
# # for outcome, hatch in zip(outcomes, hatch_patterns):
# #     offset = width * multiplier
# #     rects = ax.bar(x + offset, [outcome_counts[folder][outcome] for folder in folders], width, label=f'Outcome {outcome}', hatch=hatch)
# #     ax.bar_label(rects, padding=3)
# #     multiplier += 1
# #
# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('Count')
# # ax.set_title('Outcome Counts by Folder')
# # ax.set_xticks(x + width * (len(outcomes) - 1) / 2, folders)
# # ax.legend(loc='upper right', ncols=len(outcomes))
# # ax.set_ylim(0, max(max(outcome_counts[folder].values()) for folder in folders) * 1.1)
# #
# # plt.show()
#
# #############

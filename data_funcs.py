import os
import numpy as np
import pandas as pd
from scipy import interpolate


from classifier.helper_code import *


def get_data(data_directory, training=True):
	# Returns the X, Z and y data for this dataset

    data = [] 
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

    for i in range(num_recordings):
        if i%100 ==1:   
            print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        # Get age, sex and correctly ordered leads
        # twelve_leads here is from helper_code.py
        age, sex, recording = get_features(header, recording, twelve_leads)
        data.append(recording)
    
        # One hot encode based on header file and position in classes list
        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1

    X = np.stack(data)
    y = labels
    print('Recordings shape:', X.shape, y.shape)
    return X, y



def recordings_to_keep(header_files, recording_files, data_directory, training):
	# Returns the indexes of the recordings to keep based on training and dataset
	# Get the folder name of the dataset being processed
	dataset = data_directory.split('/')[-1]
	print('Processing:', dataset)

	if dataset == 'WFDB_PTBXL':
		# Load csv file
		path = os.path.join(sys.path[0], 'ptbxl_database.csv')
		Y = pd.read_csv(path, index_col='ecg_id')
		if training:
			to_keep = np.where(Y.strat_fold < 9)
		else:
			to_keep = np.where(Y.strat_fold >= 9)

		header_files = header_files[to_keep]
		recording_files = recording_files[to_keep]


	return header_files, recording_files


def get_classes():
    path_to_classes = os.path.join(sys.path[0], 'evaluation/dx_mapping_scored.csv')
    df = pd.read_csv('/content/HeartbeatClassification/evaluation/dx_mapping_scored.csv')
    SNOMED_CT_Codes = list(df['SNOMED CT Code'])
    class_abbreviations = list(df['Abbreviation'])
    return SNOMED_CT_Codes, class_abbreviations


def one_hot_encode_labels(header, classes):
	#####
	# need to ensure the header label and class labels are both of same data type (cant mix string and int)
	num_classes = len(classes)
	labels = np.zeros(num_classes, dtype=np.bool) # One-hot encoding of classes
    current_labels = get_labels(header)
    for label in current_labels:
        if label in classes:
            j = classes.index(label)
            labels[j] = 1
	return labels


# downsample to 100hz for the model to accept
def downsample_recording(recording, frequency, num_samples):
	ecg_len = num_samples/frequency  # Length of time in seconds
	t = np.linspace(0, ecg_len, num=num_samples)
	# Create an interpolation class to return the values at a given t_new
	interp_func = interpolate.interp1d(x, recording, kind='linear', axis=0, assume_sorted=True)

	freq_new = 100
	num_samples_new = ecg_len*freq_new
	# New times to evaluate function at (same length but fewer samples)
	t_new = np.linspace(0, ecg_len, num=num_samples_new)
	recording_new = interp_func(t_new)
	return recording_new


















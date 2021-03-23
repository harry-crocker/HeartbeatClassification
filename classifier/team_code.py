#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from ..model_funcs import *
from ..data_funcs import *

# To change when found save model
twelve_lead_model_filename = '12_lead_model'
six_lead_model_filename = '6_lead_model'
three_lead_model_filename = '3_lead_model'
two_lead_model_filename = '2_lead_model'

################################################################################
#
# Training function
#
################################################################################


# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dx_mapping_scored.csv file as want to have same classes for all models
    print('Extracting classes...')
    classes, abb_classes = get_classes()
    num_classes = len(classes)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')
    X = []
    y = []

    # In the real submission all training files are in a single folder
    header_files, recording_files = find_challenge_files(data_directory)
    # Drop files in train/test split
    # header_files, recording_files = recordings_to_keep(header_files, recording_files, data_directory, training)
    num_recordings = len(recording_files)
    print(num_recordings, 'Files found')
    if not num_recordings:
        raise Exception('No data within:', data_directory.split('/')[-1])



    # Load model configuration file

    # run = wandb.init(project='HeartbeatClassification')
    # config = wandb.config
    class Config_file():
        pass

    config = Config_file()

    config.num_modules = 6 # 6
    config.epochs = 2 # PTB-XL = 50
    config.lr = 3e-3  # 1e-2
    config.batch_size = 32  # PTB-XL = 128
    config.ctype = 'subdiagnostic'
    config.optimizer='AdamWeightDecay'
    config.wd = 1e-2 # Float
    config.Window_length = 125 # 250
    config.loss_func = 'BC'   # BC Or F1
    config.SpE = 1 # 1
    config.filters = 32
    config.kernel_sizes = [9, 23, 49]
    config.head_nodes = 2048

    input_shape = [config.Window_length, 12]
    lap = 0.5

    cbs = []
    # cbs.append(WandbCallback())
    # LR Schedule callback 
    steps = config.SpE * np.ceil(len(recording_files) / config.batch_size) * config.epochs
    lr_schedule = OneCycleScheduler(config.lr, steps, wd=config.wd, mom_min=0.85, mom_max=0.95)
    cbs.append(lr_schedule)
        
    # Build Model
    model = Build_InceptionTime(input_shape, num_classes, config.num_modules, config.lr, config.wd, config.optimizer, config.loss_func, 
                                config.Window_length, lap, config.filters, config.kernel_sizes, config.head_nodes)

    # Train model
    history = model.fit(train_generator(header_files, recording_files, config.Window_length, config.batch_size), 
                    steps_per_epoch= steps // config.epochs,
                    epochs=config.epochs, 
                    batch_size=config.batch_size, 
                    # validation_data=(X_val, y_val),
                    callbacks=cbs)

    # Save model
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    model.save(filename)

    # Train models


    ### Adapt this section to only train the number of leads we want rather than all 4 models every time
    # Train 12-lead ECG model.
    # print('Training 12-lead ECG model...')

    # leads = twelve_leads
    # filename = os.path.join(model_directory, twelve_lead_model_filename)

    # # Chooses which values of data to use where [12, 13] are age and sex
    # # Feed this into generator for extracting correct leads
    # # Will also need to ensure that when loading recording that leads are in correct order
    # feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    # features = data[:, feature_indices]

    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    # classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    # #### Need to investigate this function
    # save_model(filename, classes, leads, imputer, classifier)



################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
## Dont use this
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    return tf.keras.models.load_model(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
## Change this to accept a single recording 
## Classes should use my function get_classes()
def run_model(model, header, recording):
    # Preprocess recording
    recording = np.swapaxes(recording, 0, 1)    # Needs to be of form (num_samples, num_channels)
    # Get sampling data from header
    frequency = get_frequency(header)
    num_samples = get_num_samples(header)
    # Downsample the recording
    recording = downsample_recording(recording, frequency, num_samples)
    recording = np.expand_dims(recording, 0)    # Needs to be of form (num_recordings, num_samples, num_channels)

    classes, _ = get_classes()

    # Predict
    outputs = model.compute_predictions(recording)

    # Predict labels and probabilities.
    thresh = 0.4        # Could find better thresholds and load them above from the model directory by appending 'benchmarks' to the end of filename
    labels = np.where(outputs > thresh, 1, 0)
    labels = list(labels[0])

    probabilities = list(outputs[0])

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
### Can use age and sex extraction from this but dont want the final rms part
def get_features(header, recording, leads, preprocessing=False):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    if preprocessing:
        adc_gains = get_adcgains(header, leads)
        baselines = get_baselines(header, leads)
        num_leads = len(leads)
        for i in range(num_leads):
            recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    return age, sex, recording

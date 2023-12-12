import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, lfilter

def loadTrainingData():
    """
    Load the data from the D1.mat file.
    
    :return: d, index, label
    """


    # Get the path to the data
    dataset_path = "./data/D1.mat"

    # load the data
    data = sio.loadmat(dataset_path, squeeze_me=True)

    # d is the raw time domain recording (1,440,000) 25kHz samlping frequency
    d = data.get('d')
    # index is the location in the recording of the start of each spike
    index = data.get('Index')
    # label is the class of each spike (1, 2, 3, 4, or 5), i.e. the type of
    # neuron that fired it
    label = data.get('Class')

    # return the data
    return d, index, label

def loadPredictionData(filepath):
    """
    Load the data from the given filepath.

    :param filepath: filepath
    
    :return: d
    """

    # load the data
    data = sio.loadmat(filepath, squeeze_me=True)

    # d is the raw time domain recording (1,440,000) 25kHz samlping frequency
    d = data.get('d')

    # return the data
    return d

def savePredictions(filepath, predictions, predictions_indicies):
    """
    Save the predictions to the given filepath.

    :param filepath: filepath
    :param predictions: predictions
    :param predictions_indicies: predictions indicies
    
    :return: None
    """

    # save the predictions as a .mat file
    sio.savemat(filepath, {'Class': predictions, 'Index': predictions_indicies})

def preprocessTrainingData(d, index, label, cutoff_freq=1000, sampling_freq=25000, window_size=100):
    """
    Preprocess the data
    
    :param d: the raw time domain recording
    :param index: the locations in the recording of the start of each spike
    :param label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of
                    neuron that fired it
    :param cutoff_freq: cutoff frequency
    :param sampling_freq: sampling frequency
    :param window_size: window size

    :return: df
    """

    
    df = createDataFrame(d, index, label)

    # normalize the data
    df_norm = normalizeAmplitudes(df)

    # filter the data
    df_filtered = lowPassFilter(df_norm, cutoff_freq, sampling_freq)

    # Split the data into windows
    df_windows = createScanningWindows(df_filtered, window_size)

    # Unbias the data
    df_unbias = unbiasData(df_windows)

    return df_unbias

def preprocessPredictionData(d, cutoff_freq=1000, sampling_freq=25000, window_size=100):
    """
    Preprocess the data
    
    :param d: the raw time domain recording
    :param cutoff_freq: cutoff frequency
    :param sampling_freq: sampling frequency
    :param window_size: window size

    :return: df
    """
    df = createDataFrame(d)

    # normalize the data
    df_norm = normalizeAmplitudes(df)

    # filter the data
    df_filtered = lowPassFilter(df_norm, cutoff_freq, sampling_freq)

    # Split the data into windows
    df_windows = createScanningWindows(df_filtered, window_size)

    return df_windows

def createDataFrame(d, index=None, label=None, sampling_freq=25000):
    """
    Create a dataframe from the data. The dataframe should have the following
    columns:
    - Time: the time in seconds
    - Amplitude: the amplitude of the signal
    - Label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of neuron
             that fired it
    - labeln: a one-hot encoded vector of the label column where n is the class

    :param d: the raw time domain recording
    :param index: the locations in the recording of the start of each spike
    :param label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of
                  neuron that fired it
    :param sampling_freq: sampling frequency

    :return: dataframe
    """

    # Create a dataframe which contains the time in seconds and the amplitude of
    # the signal
    df = pd.DataFrame()
    df['Time'] = np.arange(0, len(d)/sampling_freq, 1/sampling_freq)
    df['Amplitude'] = d

    # if index and label are not None then add a label column to the dataframe.
    if index is not None and label is not None:
        # add a label column to the dataframe and set all the values to 0
        df['Label'] = 0
        # set the values of the label column to the values in the label array at the
        # locations given by the index array
        df.loc[index, 'Label'] = label

        # convert the label column to a one-hot encoded vector
        labels =  tf.keras.utils.to_categorical(df['Label'])

        # add the one-hot encoded vector to the dataframe
        for i in range(len(labels[0])):
            df['Label' + str(i)] = labels[:, i]

    # return the dataframe
    return df

def normalizeAmplitudes(df):
    """
    Normalize the amplitudes so that they are between -1 and 1 by dividing the
    amplitudes by the maximum amplitude value.
    
    :param df: dataframe
    
    :return: dataframe
    """
    # normalize the amplitude column so that the values are between -1 and 1 by
    # dividing by the maximum value
    df['Amplitude'] = df['Amplitude'] / df['Amplitude'].max()

    return df

def lowPassFilter(df, cutoff_freq, sampling_freq, order=5):
    """
    Low pass filter the data
    
    :param df: dataframe
    :param cutoff_freq: cutoff frequency
    :param sampling_freq: sampling frequency
    :param order: order
    
    :return: dataframe
    """

    # create a copy of the df
    df_filtered = df.copy()

    b, a = butter(order, cutoff_freq, fs=sampling_freq, btype='low', analog=False)

    # apply the filter to the amplitude column
    df_filtered['Amplitude'] = lfilter(b, a, df_filtered['Amplitude'])

    # Accommodate for the phase shift caused by the filter
    df_filtered['Amplitude'] = np.roll(df_filtered['Amplitude'], -2 * order)

    return df_filtered

def createScanningWindows(df, window_size):
    """
    Create windows of the data. This is done by creating a dataframe with the
    amplitude column duplicated window_size times. Each amplitude column is
    shifted by -1. The first column is not shifted and the last column is
    shifted by -window_size. Therefore, each row in the dataframe contains the
    amplitude values for window_size consecutive samples.
    
    :param df: dataframe
    :param window_size: window size
    
    :return: dataframe
    """

    # create a dataframe with the amplitude column duplicated window_size times
    # and include a column for the time and label values.
    df_windows = pd.concat([
        df['Amplitude'].shift(-i, fill_value=0) for i in range(window_size)
    ], axis=1)
    df_windows.columns = [f'Amplitude{i}' for i in range(window_size)]
    
    # add the time and label columns to the dataframe
    column_names = df.columns.difference(['Amplitude'])
    # concat the df_windows and df[column_names] dataframes
    df_windows = pd.concat([df_windows, df[column_names]], axis=1)

    return df_windows

def unbiasData(df):
    """
    Unbias the data by keeping the number of windows for each label the same.
    
    :param df: dataframe
    
    :return: dataframe
    """

    # Get the columns that start with 'Label' and suffix with a number
    label_names = df.filter(regex='Label\d+').columns
    # for each label column, sum the values in the column
    label_sums = df[label_names].sum()
    # get the minimum sum
    min_sum = label_sums.min()
    # Randomly select min_sum rows from each label
    df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(int(min_sum)))
    
    # reset the index
    df = df.reset_index(drop=True)
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def getTrainAndTestData(df, train_size):
    """
    Split the data into training and test sets.

    :param df: dataframe
    :param train_size: Percentage of the data to use for training.

    :return: training and test sets
    """

    # Get the number of rows in the dataframe
    n_rows = df.shape[0]

    # Get the number of rows in the training set
    n_train = int(train_size * n_rows)

    # Split the dataframe into training and test sets
    df_train = df.iloc[:n_train, :]
    df_test = df.iloc[n_train:, :]

    return df_train, df_test




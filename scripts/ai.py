import pandas as pd
import numpy as np
import tensorflow as tf

from scipy.signal import butter, lfilter

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

def preprocessData(d, index, label):
    """
    Preprocess the data

    :param d: the raw time domain recording
    :param index: the locations in the recording of the start of each spike
    :param label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of neuron that fired it

    :return: dataframe
    """

    # sampling frequency
    sampling_freq = 25000

    # Create a dataframe which contains the time in seconds and the amplitude of
    # the signal
    df = pd.DataFrame()
    df['Time'] = np.arange(0, len(d)/sampling_freq, 1/sampling_freq)
    df['Amplitude'] = d

    # add a label column to the dataframe
    df['Label'] = 0
    df.loc[index, 'Label'] = label

    # convert the label column to categorical
    df['Label'] = pd.Categorical(df['Label'])

    # use one-hot encoding to create dummy variables for the label column
    df_label = pd.get_dummies(df['Label'], prefix='Label')

    # concatenate the dummy variables to the dataframe
    df = pd.concat([df, df_label], axis=1)

    # drop the original label column
    df = df.drop('Label', axis=1)

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

    b, a = butter(order, cutoff_freq, fs=sampling_freq, btype='low', analog=False)

    # apply the filter to the amplitude column
    df['Amplitude'] = lfilter(b, a, df['Amplitude'])

    # Accommodate for the phase shift caused by the filter
    df['Amplitude'] = np.roll(df['Amplitude'], -2* order)

    return df

def splitData(df, train_size):
    """
    Split the data into training and test sets

    :param df: dataframe
    :param train_size: training set size {0, 1}

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

def generateSpectogram(df, frame_length, frame_step):
    """
    Generate a spectogram of the data
    
    :param df: dataframe
    :param frame_length: frame length
    :param frame_step: frame step
    
    :return: spectogram
    """
    
    # get the amplitude column
    amplitude = df['Amplitude'].values
    
    # get the time values
    times = df['Time'].values
    
    # get the spectrogram
    spectogram = tf.signal.stft(amplitude, frame_length=frame_length, frame_step=frame_step)
    spectogram = tf.abs(spectogram)
    spectogram = tf.expand_dims(spectogram, axis=2)
    
    return spectogram


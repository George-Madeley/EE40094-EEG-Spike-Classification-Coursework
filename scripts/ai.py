import pandas as pd
import numpy as np
import tensorflow as tf

from scipy.signal import butter, lfilter

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer
from keras.optimizers import Adam

import matplotlib.pyplot as plt

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
    df_filtered['Amplitude'] = np.roll(df_filtered['Amplitude'], -2* order)

    return df_filtered

def splitIntoWindows(df, window_size):
    """
    Split the data into windows. The last windows are padded with zeros if
    necessary.
    
    :param df: dataframe
    :param window_size: window size
    
    :return: dataframe
    """

    # create a dataframe with the amplitude column duplicated window_size times
    # and include a column for the time and label values.
    df_windows = pd.DataFrame()
    for i in range(window_size):
        df_windows['Amplitude' + str(i)] = df['Amplitude'].shift(-i, fill_value=0)
    
    df_windows['Time'] = df['Time']
    df_windows['Label'] = df['Label']

    return df_windows


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

    # plot the spectrogram
    plt.figure(figsize=(10, 8))
    plt.imshow(tf.transpose(spectogram)[0])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.show()
    
    return spectogram

def train(train_df, window_size, step_size, batch_size, epochs, num_neurons):
    """
    Train the model
    
    :param train_df: training dataframe
    :param window_size: window size
    :param step_size: step size
    :param batch_size: batch size
    :param epochs: epochs
    
    :return: model
    """

    # get the amplitude column
    amplitude = train_df['Amplitude'].values

    # get the time values
    times = train_df['Time'].values

    # Create a sequential neural network model with 1 input layer, 2 hidden
    # layers and 1 output layer. window_size is the number of input neurons.
    # num_neurons is the number of neurons in the hidden layers.
    model = Sequential()
    model.add(InputLayer(input_shape=(window_size,)))




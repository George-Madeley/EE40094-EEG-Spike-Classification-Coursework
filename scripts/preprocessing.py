import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.io as sio
import sys
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter



def loadData():
    """
    Load the data from the .mat file specified by the command line argument.
    
    :return: d, index, label
    """

    # Check if there is a command line argument
    if len(sys.argv) != 2:
        # If there is no command line argument, print the usage message and exit
        print("Usage: python main.py <path_to_data>")
        sys.exit()

    # Get the path to the data
    dataset_path = sys.argv[1]

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

def createDataFrame(d, index, label, sampling_freq=25000):
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

def createWindows(df, window_size):
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
    df_windows = pd.DataFrame()
    for i in range(window_size):
        df_windows['Amplitude' + str(i)] = df['Amplitude'].shift(-i, fill_value=0)
    
    # add the time and label columns to the dataframe
    column_names = []
    for column_name in df.columns:
        if column_name != 'Amplitude':
            column_names.append(column_name)
    # concat the df_windows and df[column_names] dataframes
    df_windows = pd.concat([df_windows, df[column_names]], axis=1)

    return df_windows

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

def plot_data(index, label, df_norm, df_filtered, num_samples_plot=5000):
    """
    Plot the raw and filtered data
    
    :param index: the locations in the recording of the start of each spike
    :param label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of
                  neuron that fired it
    :param df_norm: the normalized dataframe
    :param df_filtered: the filtered dataframe
    
    :return: None
    """
    plt.plot(df_norm['Time'][:num_samples_plot],
             df_norm['Amplitude'][:num_samples_plot],
             color='blue',
             label='Raw')

    # Plot the locations of the spikes on to the graph given by the index array.
    # The marker should be an number given by the label array.
    indices_plot = index[index < num_samples_plot]
    labels_plot = label[index < num_samples_plot]
    plt.scatter(
        df_norm['Time'][indices_plot],
        df_norm['Amplitude'][indices_plot],
        c=labels_plot,
        cmap='rainbow',
        label='Spikes',
        s=100,
        marker='x'
    )

    
    # plot the filtered data on to a line graph
    plt.plot(df_filtered['Time'][:num_samples_plot],
             df_filtered['Amplitude'][:num_samples_plot],
             color='red',
             label='Filtered')

    # add a legend to the graph
    plt.legend()
    # add a title to the graph
    plt.title('Raw vs Filtered Data')
    # add labels to the x and y axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    # display the graph
    plt.show()

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

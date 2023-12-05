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

    dataFrames = []
    for SNR in range (0, 101, 20):
        # add noise to the data
        df_noise = addNoise(df_norm, SNR)
        # filter the data
        df_filtered = lowPassFilter(df_noise, cutoff_freq, sampling_freq)
        # Split the data into windows
        df_windows = createWindows(df_filtered, window_size)
        # Unbias the data
        df_unbias = unbiasData(df_windows)
        # add the dataframe to the list of dataframes
        dataFrames.append(df_unbias)

    # concat the dataframes
    df = pd.concat(dataFrames)

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df

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
    df_windows = createWindows(df_filtered, window_size)

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

def addNoise(df, SNR):
    """
    Add noise to the data
    
    :param df: dataframe
    :param SNR: signal to noise ratio
    
    :return: dataframe

    :raises ValueError: if SNR is not between 0 and 100
    """

    if 0 > SNR or SNR > 100:
        raise ValueError('SNR must be between 0 and 100')

    # Generate a noisy signal ranging from -1 to 1
    noise = np.random.uniform(-1, 1, len(df['Amplitude']))

    # Divide the SNR by 100 to get the difference between the max amplitude and
    # the max noise
    SNR = SNR / 100

    # Get the max noise amplitude
    noise_factor = 1 - SNR

    # Multiply the noise by the noise factor
    noise = noise * noise_factor

    df_noise = df.copy()

    # Add the noise to the amplitude column
    df_noise['Amplitude'] = df_noise['Amplitude'] + noise

    # Normalize the amplitude column so that the values are between -1 and 1
    df_noise = normalizeAmplitudes(df)

    return df_noise

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

def plot_data(df, num_samples_plot=5000, color='blue', label='Raw'):
    """
    Plot the raw and filtered data
    
    :param df: dataframe
    :param num_samples_plot: number of samples to plot
    :param color: color
    :param label: label
    
    :return: None
    """
    # plot the data
    plt.plot(df['Time'][:num_samples_plot],
             df['Amplitude'][:num_samples_plot],
             color=color,
             label=label)

    # add a legend to the graph
    plt.legend()
    # add a title to the graph
    plt.title('Raw vs Filtered Data')
    # add labels to the x and y axes
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

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

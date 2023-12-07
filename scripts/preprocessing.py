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

def preprocessTrainingData(d, index, label, low_cutoff_freq=1000, high_cutoff_freq=1000, sampling_freq=25000, window_size=100, zero_bias_coefficient=8):
    """
    Preprocess the data
    
    :param d: the raw time domain recording
    :param index: the locations in the recording of the start of each spike
    :param label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of
                    neuron that fired it
    :param low_cutoff_freq: cutoff frequency for the low pass filter
    :param high_cutoff_freq: cutoff frequency for the high pass filter
    :param sampling_freq: sampling frequency
    :param window_size: window size
    :param zero_bias_coefficient: If the coefficient is 1, then the number of
                                    windows for the negative label will be the
                                    same as the number of windows for the
                                    positive labels. If the coefficient is
                                    greater than 1, then the number of windows
                                    for the negative label will be greater than
                                    the number of windows for the positive and
                                    vice versa.

    :return: df
    """

    
    df = createDataFrame(d, index, label)

    # normalize the data
    df_norm = normalizeAmplitudes(df)

    dataFrames = []
    low_noise = 20
    high_noise = 100
    step = int((high_noise - low_noise) / 5)
    for SNR in range(low_noise, high_noise + 1, step):
        # add noise to the data
        df_noise = addNoise(df_norm, SNR)
        # filter the data. Filtering can cause the amplitudes to decrease in
        # power so the data is normalized again.
        df_low_filtered = lowPassFilter(df_noise, low_cutoff_freq, sampling_freq)
        df_high_filtered = highPassFilter(df_low_filtered, high_cutoff_freq, sampling_freq)
        df_filtered = normalizeAmplitudes(df_high_filtered)
        # Split the data into windows
        df_windows = createWindows(df_filtered, window_size)
        # Unbias the data
        df_unbias = unbiasData(df_windows, zero_bias_coefficient)
        # add the dataframe to the list of dataframes
        dataFrames.append(df_unbias)

    # concat the dataframes
    df = pd.concat(dataFrames)

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def preprocessPredictionData(d, low_cutoff_freq=1000, high_cutoff_freq=1000, sampling_freq=25000, window_size=100):
    """
    Preprocess the data
    
    :param d: the raw time domain recording
    :param low_cutoff_freq: cutoff frequency for the low pass filter
    :param high_cutoff_freq: cutoff frequency for the high pass filter
    :param sampling_freq: sampling frequency
    :param window_size: window size

    :return: df
    """
    df = createDataFrame(d)

    # normalize the data
    df_norm = normalizeAmplitudes(df)

    # filter the data. Filtering can cause the amplitudes to decrease in
    # power so the data is normalized again.
    df_low_filtered = lowPassFilter(df_norm, low_cutoff_freq, sampling_freq)
    df_high_filtered = highPassFilter(df_low_filtered, high_cutoff_freq, sampling_freq)
    df_filtered = normalizeAmplitudes(df_high_filtered)

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

    # Generate a gaussian noise signal with amplitudes ranging from -1 to 1 and
    # a mean of 0 and a standard deviation of 1 and frequencies from 0 to 25kHz
    # with a sampling frequency of 25kHz
    noise = np.random.normal(0, 1, len(df['Amplitude']))

    # Normalize the noise so that the values are between -1 and 1
    noise = noise / noise.max()

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

    # Cap the amplitude values at -1 and 1
    df_noise['Amplitude'] = df_noise['Amplitude'].clip(-1, 1)

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

def highPassFilter(df, cutoff_freq, sampling_freq, order=5):
    """
    High pass filter the data
    
    :param df: dataframe
    :param cutoff_freq: cutoff frequency
    :param sampling_freq: sampling frequency
    :param order: order

    :return: dataframe
    """

    # create a copy of the df
    df_filtered = df.copy()

    # Create the filter
    b, a = butter(order, cutoff_freq, fs=sampling_freq, btype='high', analog=False)

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

def unbiasData(df, zero_bias_coefficient=8):
    """
    Unbias the data by keeping the number of windows for each label the same.
    
    :param df: dataframe
    :param zero_bias_coefficient: If the coefficient is 1, then the number of
                                    windows for the negative label will be the
                                    same as the number of windows for the
                                    positive labels. If the coefficient is
                                    greater than 1, then the number of windows
                                    for the negative label will be greater than
                                    the number of windows for the positive and
                                    vice versa.
    
    :return: dataframe
    """

    # To unbias the data, we need to ensure that all positive labels (i.e., 
    # labels that are not 0) have the same number of windows. This ensures there
    # is no bias towards a particular label. However, we also want to keep the
    # number of windows for the negative label (i.e., label 0) the same as the
    # total number of windows for the positive labels. This ensures that the
    # model does not over predict the positive labels.

    # Before either of these can be done, we need to find the minimum number of
    # windows for the positive labels. This will be used as the number of
    # windows for the positive labels and the negative label.

    # We first get all of the one-hot encoded label columns and sum the values
    # in each column. This will give us the number of windows for each label. We
    # then find the minimum number of windows for the positive labels.
    label_names = df.filter(regex='Label\d+').columns
    label_sums = df[label_names].sum()
    min_sum = int(label_sums.min())

    # To unbias the data, we group the rows by the label column. We then
    # randomly select min_sum  number of rows from each group. This ensures that
    # the number of windows for each label is the same.
    grouped = df.groupby('Label')
    df_group = grouped.apply(lambda x: x.sample(min_sum)).reset_index(drop=True)

    # However, this will result in the number of windows for the negative label
    # being less than the number of windows for the positive labels. To fix
    # this, we remove all the rows that have a label of 0. We then randomly
    # select (min_sum * number of positive labels) number of rows from the
    # dataframe where the label is 0.
    df_group = df_group[df_group['Label'] != 0]
    # calculate the number of lables
    num_labels = int(len(label_names))
    # randomly select (min_sum * number of positive labels) number of rows from
    # the dataframe where the label is 0. We can multiply the zero_bias_coefficient
    # to get a bias towards the negative label.
    # Calculate the number of rows with a label of 0
    num_label0 = df[df['Label'] == 0].shape[0]
    num_0_samples = min_sum * (num_labels - 1) * zero_bias_coefficient
    # Riase an error if the number of rows with a label of 0 is less than the
    # number of negative levels to sample
    if num_label0 < num_0_samples:
        raise ValueError('The number of samples with a label of 0 is less than the number of negative levels to sample')
    df_label0 = df[df['Label'] == 0].sample(min_sum * (num_labels - 1) * zero_bias_coefficient)

    # We then concat the two dataframes together to get the unbias dataframe.
    df = pd.concat([df_group, df_label0])
    
    # We then shuffle the dataframe to ensure that the data is not ordered by
    # label and therefore does not overfit to a particular label.
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

def plotTimeDomain(df, num_samples_plot=5000, label='Raw'):
    """
    Plot the data in the time domain
    
    :param df: dataframe
    :param num_samples_plot: number of samples to plot
    :param label: label
    
    :return: None
    """

    amplitudes = df['Amplitude'].values
    time = df['Time'].values

    # plot the data
    plt.plot(time[:num_samples_plot], amplitudes[:num_samples_plot], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Time Domain')
    
def plotFrequencyDomain(df, sampling_freq=25000, num_samples_plot=5000, label='Raw'):
    """
    PLot the data in the frequency domain
    
    :param df: dataframe
    :param sampling_freq: sampling frequency
    
    :return: None
    """

    # get the number of samples
    n = len(df['Amplitude'])

    # get the fft of the amplitude column
    fft = np.fft.fft(df['Amplitude'])

    # get the frequency values
    freq = np.fft.fftfreq(n, 1/sampling_freq)

    # frequency values are symmetrical, so we only need to plot the first half
    # of the values. In addition, we only need to plot the positive values.
    freq = freq[:n//2]
    fft = fft[:n//2]

    # remove the DC component of the fft
    fft[0] = 0

    # plot the data
    plt.plot(freq[:num_samples_plot], np.abs(fft)[:num_samples_plot], label=label)

    # add a legend to the graph
    plt.legend()
    # add a title to the graph
    plt.title('Frequency Domain')
    # add labels to the x and y axes
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.plot()


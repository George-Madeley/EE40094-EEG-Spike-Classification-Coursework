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

def preprocessTrainingData(d, index, label, low_cutoff_freq=1000, high_cutoff_freq=1000, sampling_freq=25000, window_size=100, noisePower=0):
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
    for noiseFactors in [0, noisePower]:
        # add noise to the data
        df_noise = addNoise(df_norm, noiseFactors)
        # filter the data. Filtering can cause the amplitudes to decrease in
        # power so the data is normalized again.
        df_low_filtered = lowPassFilter(df_noise, low_cutoff_freq, sampling_freq)
        # df_high_filtered = highPassFilter(df_low_filtered, high_cutoff_freq, sampling_freq)
        df_filtered = normalizeAmplitudes(df_low_filtered)
        # Split the data into windows
        df_windows = createWindows(df_filtered, window_size)

        df_noDuplicates = filterDuplicates(df_windows)

        df_sorted = sortWindows(df_noDuplicates, window_size)

        plotWindows(df_sorted, window_size)

        # Unbias the data
        df_unbias = unbiasData(df_sorted)

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
    # df_high_filtered = highPassFilter(df_low_filtered, high_cutoff_freq, sampling_freq)
    df_filtered = normalizeAmplitudes(df_low_filtered)

    # Split the data into windows
    df_windows = createWindows(df_filtered, window_size)

    # There are a bunch of duplicate rows in the dataframe. We need to remove
    # these duplicates to get the windows around the peaks. To do this, we
    # remove the rows where peakIndex is 0.
    df_windows = df_windows[df_windows['PeakIndex'] != 0]
    # We then drop the PeakIndex column
    df_windows = df_windows.drop(columns=['PeakIndex'])
    # We then remove any remaining duplicates specified by the
    # 'RelativePeakIndex' column.
    df_windows = df_windows.drop_duplicates(subset=['RelativePeakIndex'])

    # Sort the windows by the amplitude of the peak relative to the window.
    df_sorted = sortWindows(df_windows, window_size)

    return df_sorted

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

def addNoise(df, noisePower):
    """
    Add noise to the data
    
    :param df: dataframe
    :param noisePower: Percentage of the maximum amplitude value to add as noise
    
    :return: dataframe

    :raises ValueError: if SNR is not between 0 and 100
    """

    if 0 > noisePower or noisePower > 100:
        raise ValueError('Noise Power must be between 0 and 100')

    # Generate a gaussian noise signal with amplitudes ranging from -1 to 1 and
    # a mean of 0 and a standard deviation of 1 and frequencies from 0 to 25kHz
    # with a sampling frequency of 25kHz
    noise = np.random.normal(0, 1, len(df['Amplitude']))

    # Normalize the noise so that the values are between -1 and 1
    noise = noise / noise.max()

    noiseFactor = noisePower / 100

    # Multiply the noise by the noise factor
    noise = noise * noiseFactor

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
    Create windows around the peaks in the data.
    
    :param df: dataframe
    :param window_size: window size
    
    :return: dataframe
    """

    # We are going to create windows around the peaks. The radius of the windows
    # is window_size. To find these peaks, we will create a search window from
    # the index to the index + window_size. We will then find the index of the
    # first peak in that window. We will then create a window around that peak. 

    # Get the indicies of the df
    indicies = df.index

    # if the label column exists, get the indicies where the label is not 0
    if 'Label' in df.columns:
        indicies_of_labels = indicies[df['Label'] != 0]

    # Create a dataframe with the amplitude column duplicated window_size and
    # shifted by 1. This creates the search windows for the peaks.
    df_windows = pd.concat([
        df['Amplitude'].shift(-i) for i in range(window_size)
    ], axis=1)

    # When we shift the amplitude column, we get a bunch of NaN values. We need
    # to fill these values with the last value in the amplitude column. We do
    # this by using the fill_value parameter. This ensures that no peaks are
    # detected beyond the amplitude column.
    df_windows = df_windows.ffill(axis=1)

    # Rename the 'Amplitude' columns to i where i is the amount the column was 
    # shifted by
    df_windows.columns = [str(i) for i in range(window_size)]

    # Create a column which will store the index of the peak. We will use this
    # column to create the windows around the peaks. We will then drop this
    # column. The initial value of the column will be 0. If a peak is found,
    # then the value will be set to the index of the peak and the window will be
    # created around the peak. Else, the value will remain 0 and the window will
    # be created around the index or can be dropped.
    window_peak_indicies = np.zeros(len(df_windows))

    # Find the first peak in the window that is larger than the peak threshold.
    # To do this, we duplicate the amplitude columns and shift them by -1 and 1.
    # We then find the indicies where the amplitude is greater than the amplitude
    # of the previous row. We then get the first index of the indicies. This is
    # the index of the first peak in the window. We then store this index in the
    # PeakIndex column.

    # Create a copy of the df_windows dataframe and shift it by -1 and 1.
    df_windows_before = df_windows.copy()
    df_windows_after = df_windows.copy()
    df_windows_before = df_windows_before.shift(1, fill_value=0, axis=1)
    df_windows_after = df_windows_after.shift(-1, fill_value=0, axis=1)

    # Set the first value of df_windows_before to the first value of df_windows
    # and the last value of df_windows_after to the last value of df_windows.
    # We do this because we do not want to unintentionally find a peak at the
    # start or end of the window.
    df_windows_before.iloc[:, 0] = df_windows.iloc[:, 0]
    df_windows_after.iloc[:, -1] = df_windows.iloc[:, -1]

    # Get a list of indicies where the amplitude is greater than the amplitude
    # of the previous row and the amplitude of the next row and is greater than
    # the peak threshold.
    highlight_peaks = np.where(
        (df_windows > df_windows_before) &
        (df_windows > df_windows_after),
        df_windows,
        df_windows * 0
    )

    # Get the absolute value of the highlight_peaks array.
    highlight_peaks_abs = np.abs(highlight_peaks)

    # Sum the rows of the numpy array then return the index of the non-zero
    # values. This gives us the indicies of the rows where a peak was found.
    peak_row_indicies = np.where(highlight_peaks_abs.sum(axis=1) > 0)[0]

    # For each row that had a non-zero sum, get the index of the first value
    # that is bigger than 0. This is the index of the peak. However, this comes
    # with an issue for the labelled data when there are overlaps in spikes.

    # Take two labels, 1 and 2 and two peaks, A and B; both of which occur 
    # after the labels but are close enough to be in the same window. Using the
    # above method, the first peak found will be A. However, label 2 will be
    # assigned to peak A as peak A is the first peak within the window of label
    # 2.

    # To fix this issue, we need to keep track of all the indicies of peak which
    # have been assigned a label. If a peak has already been assigned a label,
    # then we skip it and move on to the next peak within the search window. We
    # repeat this process until we find a peak that has not been assigned a
    # label. If all the peaks within the search window have been assigned a
    # label, then we assign the last peak to the label.
    if 'Label' in df.columns:
        # Create an array to store the indicies of the peaks that have been
        # assigned a label
        indicies_with_labels = []
        # Loop through the indicies of the rows where a peak was found
        for peak_row_index in peak_row_indicies:
            # Get the indicies of the peaks in the row relative to the search
            # window
            peak_row = np.abs(highlight_peaks[peak_row_index])
            peak_col_indicies = np.where(peak_row > 0)[0]
            # Check if the current index is in the index of a label.
            if peak_row_index in indicies_of_labels:
                # Keep track of the number of peaks that have been assigned a
                # label within the search window. We call this the error count.
                # If the error count is equal to the number of peaks in the
                # search window, then all the peaks within the search window
                # have been assigned a different label. Therefore, we assign the
                # last peak to the label. We could raise an error here but we
                # would have to handle the error in the calling function.
                # Ideally, the search window should be large enough so that this
                # does not happen.
                error_count = 0
                for peak_col_index in peak_col_indicies:
                    # Get the index of the peak relative to the dataframe and
                    # check if it has been assigned a label. If it has, then
                    # increment the error count. Else, assign the label to the
                    # peak and break out of the loop.
                    relative_peak_index = peak_col_index + peak_row_index
                    if relative_peak_index in indicies_with_labels:
                        error_count += 1
                        continue
                    else:
                        indicies_with_labels.append(relative_peak_index)
                        window_peak_indicies[peak_row_index] = peak_col_index
                        break
                # If the error count is equal to the number of peaks in the
                # search window, then all the peaks within the search window
                # have been assigned a different label. Therefore, we assign the
                # last peak to the label.
                if error_count == len(peak_col_indicies):
                    window_peak_indicies[peak_row_index] = peak_col_indicies[-1]
            # If the current index is not in the index of a label, then we
            # assign the first peak.
            else:
                window_peak_indicies[peak_row_index] = peak_col_indicies[0]
    # If the label column does not exist, then we are dealing with prediction
    # data and therefore do not need to worry about the issue above.
    else:
        window_peak_indicies[peak_row_indicies] = np.argmax(highlight_peaks[peak_row_indicies] > 0, axis=1)

    # Perform the element wise addition of the indicies and the peak indicies.
    # This gets us the indicies of the peaks relative to the dataframe rather
    # than the search window.
    relative_peak_indicies = indicies + window_peak_indicies

    # Set the datatype of the relative_peak_indicies array to int
    relative_peak_indicies = relative_peak_indicies.astype(int)

    # Create a dataframe of the windows are each index. We do this by creating a
    # dataframe with the amplitude column duplicated 2 * window_size and shifted
    # by -window_size to window_size. This creates the windows around the
    # indicies
    df_windows = pd.concat([
        df['Amplitude'].shift(i, fill_value=0) for i in range(window_size, -window_size, -1)
    ], axis=1)

    # Rename the 'Amplitude' columns
    df_windows.columns = [f'Amplitude{i}' for i in range(2 * window_size)]

    # To only get the windows around the peaks, we use the relative_peak_indicies
    # array to get the rows of the dataframe that contain the windows around the
    # peaks.
    df_windows = df_windows.loc[relative_peak_indicies, :]

    # reset the indicies of the dataframe but keep the old indicies as a column
    # in the dataframe and name the column 'RelativePeakIndex'. This is the 
    # index position of where the peak was found relative to the dataframe.
    df_windows = df_windows.reset_index().rename(columns={'index': 'RelativePeakIndex'})

    # Add the columns that do not have the name 'Amplitude' to the dataframe
    # df_windows such as the Time column and the one-hot encoded label columns.
    df_windows = pd.concat([df_windows, df[df.columns.difference(['Amplitude'])]], axis=1)

    # Add the PeakIndex column to the dataframe df_windows and set the values to
    # the values in the window_peak_indicies array.
    df_windows['PeakIndex'] = window_peak_indicies

    # df_windows is filled with duplicate rows. We need to remove the duplicates
    # to get the windows around the peaks. However, which ones to remove is
    # dependent on whether the data is used for training or predicting. If the
    # data is used for training, then we want to remove the rows where the label
    # is 0. If the data is used for predicting, then we want to remove the rows
    # where PeakIndex is 0. This gets us only the windows around peaks. There is
    # still a chance there are duplicates so we just remove any duplicate
    # records This is because the data used for predicting does not have a label
    # column.

    return df_windows

def filterDuplicates(df_windows):
    """
    Filter the duplicates in the dataframe.
    
    :param df_windows: dataframe
    
    :return: dataframe
    """
    df_null = df_windows[df_windows['Label'] == 0]
    df_not_null = df_windows[df_windows['Label'] != 0]

        # We then remove the rows where there is no peak. This is done by
        # removing the rows where PeakIndex is 0.
    df_null = df_null[df_null['PeakIndex'] != 0]
        # We then drop the PeakIndex column
    df_null = df_null.drop(columns=['PeakIndex'])
        # We then remove any remaining duplicates specified by the
        # 'RelativePeakIndex' column.
    df_null = df_null.drop_duplicates(subset=['RelativePeakIndex'])

        # There is still a chance that some null labels window are duplicates of
        # the not null label windows. To remove these duplicates, we get the
        # 'RelativePeakIndex' column of the df_not_null dataframe and remove any
        # rows in the df_null dataframe that have the same value in the
        # 'RelativePeakIndex' column.
    relativePeakIndicies = df_not_null['RelativePeakIndex'].values
    df_null = df_null[~df_null['RelativePeakIndex'].isin(relativePeakIndicies)]

        # We then get the rows where the label is not 0, remove the PeakIndex
        # column, and combine with the df_null dataframe.
    df_not_null = df_not_null.drop(columns=['PeakIndex'])
    df_noDuplicates = pd.concat([df_not_null, df_null])
    return df_noDuplicates

def sortWindows(df, window_size):
    """
    Sort each window in the dataframe by the amplitude of the peak relative to
    the window.
    
    :param df: dataframe
    :param window_size: window size
    
    :return: dataframe
    """
    
    # We now need to order the windows based the height. The height of the window
    # is the difference between the peak and the minimum value in the window.
    # We then sort the windows by the height in descending order. This ensures
    # that the windows with the largest peaks are at the top of the dataframe.

    # Create a copy of the dataframe
    df_sorted = df.copy()

    # Get peak values for each row
    peak_values = df_sorted.filter(regex='Amplitude\d+').values[:, window_size]

    # Get the minimum values for each row
    min_values = df_sorted.filter(regex='Amplitude\d+').values.min(axis=1)

    # Get the height of each window
    heights = peak_values - min_values

    # Add the height column to the dataframe
    df_sorted['Height'] = heights

    # Sort the dataframe by the height column in descending order
    df_sorted = df_sorted.sort_values(by='Height', ascending=False)

    return df_sorted

def unbiasData(df):
    """
    Unbias the data by keeping the number of windows for each label the same.
    
    :param df: dataframe
    
    :return: dataframe
    """

    # To unbias the data, we need to ensure that all labels have the same number
    # of windows. This ensures there is no bias towards a particular label.

    # Before either of these can be done, we need to find the minimum number of
    # windows for the labels.

    # We first get all of the one-hot encoded label columns and sum the values
    # in each column. This will give us the number of windows for each label. We
    # then find the minimum number of windows for the labels.
    label_names = df.filter(regex='Label\d+').columns
    label_sums = df[label_names].sum()
    min_sum = int(label_sums.min())

    # To unbias the data, we group the rows by the label column. We then
    # randomly select min_sum  number of rows from each group. This ensures that
    # the number of windows for each label is the same.
    grouped = df.groupby('Label')
    df = grouped.apply(lambda x: x.sample(min_sum)).reset_index(drop=True)
    
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

def plotWindows(df, window_size):
    """
    Plot the windows around the peaks.
    
    :param df: dataframe
    :param window_size: window size
    
    :return: None
    """
    plt.subplots(2, 3)
        # group the rows by the label column
    grouped = df.groupby('Label')
        # plot the first 20 windows for each label on the same graph
    for i, (label, group) in enumerate(grouped):
            # get the values in the amplitude columns
        amplitudes = group.filter(regex='Amplitude\d+').values
        plt.subplot(2, 3, i + 1)
        # plot the amplitudes as the color grey with an alpha of 0.5
        plt.plot(np.arange(-window_size, window_size), amplitudes[:200, :].T, color='grey', alpha=0.1)
        plt.xlabel('Window Index')
        plt.ylabel('Amplitude')
        plt.title(f'Class {label}')
    plt.show()

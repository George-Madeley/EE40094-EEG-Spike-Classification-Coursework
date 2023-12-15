import pandas as pd
import numpy as np
import numpy.typing as npt
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, lfilter

from typing import Union

def preprocessData(
        filepath: str,
        peak_window_radi=(30, 30),
        search_window_size=100,
        peak_threshold=0,
        normalise_peak=False,
    ) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data by filtering, normalizing, splitting into windows, and
    removing duplicates and outliers.

    :param filepath: filepath to the prediction data
    :param peak_window_radi: The radius of the window around the peak. The first
        value is the number of samples before the peak and the second value is
        the number of samples after the peak.
    :param search_window_size: The size of the search window.
    :param peak_threshold: The minimum value that a peak can have to be
        considered a peak.
    :param normalise_peak: If True, the peaks are normalised by the maximum
        value in the windows, else the peaks are normalised by the minimum value
        in the windows.

    :return: dataframe of training data, dataframe of prediction data.
    """

    # Load the training data
    d, index, label = loadData('./data/D1.mat')
    df_train = createDataFrame(d, index, label)

    # Load the prediction data
    d = loadData(filepath)
    df_predi = createDataFrame(d)

    # filter the data
    df_train = bandPassFilter(df_train)
    df_predi = bandPassFilter(df_predi)

    # Normalize the data so that the amplitudes are between -1 and 1, keeping
    # the DC component of the signal at 0.
    df_train = normalizeMax(df_train)
    df_predi = normalizeMax(df_predi)

    # Split the data into windows around the peaks
    df_train = createWindows(
        df_train,
        peak_window_radi,
        search_window_size,
        peak_threshold=0
    )
    df_predi = createWindows(
        df_predi,
        peak_window_radi,
        search_window_size,
        peak_threshold=peak_threshold
    )

    # Remove any duplicate windows from the data
    df_train = removeDuplicates(df_train)
    df_predi = removeDuplicates(df_predi)

    # Normalise the windows so that the amplitudes are between 0 and 1
    df_train = normalizeMin(df_train, normalise_peak)
    df_predi = normalizeMin(df_predi, normalise_peak)

    # Remove any outliers from the data
    df_train = removeOutliers(df_train, peak_window_radi, threshold=0.5)
 
    # Unbias the data to prevent the model from over predicting the a particular
    # label
    df_train = unbiasData(df_train, bias_coefficients=[5, 1, 1, 1, 1, 1])

    # shuffle the dataframe to prevent the model from overfitting to a
    # particular label
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    # Return the training and prediction data
    return df_train, df_predi

def loadData(filepath: str) -> any:
    """
    Load the data from the given filepath.

    :param filepath: filepath to the data
    
    :return:
        - d: the raw time domain recording
        - index: the locations in the recording of the start of each spike
        - label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of
                    neuron that fired it in order that they appear in the data.
    """

    # load the data
    data = sio.loadmat(filepath, squeeze_me=True)

    # d is the raw time domain recording (1,440,000) 25kHz samlping frequency
    d = data.get('d')

    # index is the location in the recording of the start of each spike. If the
    # data is prediction data, then index is None.
    index = data.get('Index', None)

    # label is the class of each spike (1, 2, 3, 4, or 5), i.e. the type of
    # neuron that fired it in order that they appear in the data. If the data is
    # prediction data, then label is None.
    label = data.get('Class', None)

    # If index and label are None, then we are dealing with prediction data.
    # Therefore, we return the raw time domain recording.
    if index is None and label is None:
        return d

    # Else, we return the raw time domain recording, the index, and the label.
    return d, index, label

def createDataFrame(
        d: npt.NDArray,
        index=None,
        label=None, 
        sampling_freq=25000
    ) -> pd.DataFrame:
    """
    Create a dataframe from the data.

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

    # add a label column to the dataframe and set all the values to 0.
    df['Label'] = 0

    # if index and label are None, then we are dealing with prediction data.
    # Therefore, we return the dataframe.
    if index is None or label is None:
        return df
    
    # Else, we add labels to the dataframe at the indicies specified by the
    # index array.    
    df.loc[index, 'Label'] = label

    # convert the label column to a one-hot encoded vector.
    labels =  tf.keras.utils.to_categorical(df['Label'])

    # Add the one-hot encoded vector to the dataframe and name the columns
    # 'Labeln' where n class number.
    for i in range(len(labels[0])):
        df['Label' + str(i)] = labels[:, i]

    # return the dataframe
    return df

def bandPassFilter(df: pd.DataFrame, order=2) -> pd.DataFrame:
    """
    Band pass filter the data
    
    :param df: dataframe containing the amplitude column to filter
    :param order: order
    
    :return: dataframe
    """
    # Create the butterworth band pass filter
    b, a = butter(order, [0.005, 0.05], btype='band', analog=False)

    # Get the amplitude column as a numpy array
    amplitudes = df['Amplitude'].values

    # Apply the filter to the amplitude column and add the filtered amplitudes
    # to the dataframe
    df['Amplitude'] = lfilter(b, a, amplitudes)

    # return the dataframe
    return df

def normalizeMax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the amplitudes so that they are between -1 and 1.

    :param df: dataframe containing the amplitude column to normalize

    return df
    """
    # To normalize the amplitudes, we divide the amplitudes by the maximum
    # absolute value of the amplitudes. This ensures that the amplitudes are
    # between -1 and 1. We then store the normalized amplitudes in the amplitude
    # column.
    amplitudes = df['Amplitude'].values
    amplitudes = amplitudes / amplitudes.max()
    df['Amplitude'] = amplitudes

    return df

def normalizeMin(df: pd.DataFrame, normalise_peak: bool) -> pd.DataFrame:
    """
    Normalize the amplitudes so that they are between 0 and 1.
    
    :param df: dataframe
    :param normalise_peak: If True, the peaks are normalised by the maximum
        value in the windows, else the peaks are just normalised by the minimum
        value in the windows.
    
    :return: dataframe
    """

    # Get the the columns that start with 'Amplitude' and suffix with a number.
    # These are the columns that contain the amplitudes of the windows. We then
    # get the values of these columns as a numpy array. We then subtract the
    # minimum value in each row from the values in each row. This ensures that
    # the minimum value in each row is 0.
    amplitude_names = df.filter(regex='Amplitude\d+').columns
    amplitudes = df[amplitude_names].values
    amplitudes = amplitudes - amplitudes.min(axis=1).reshape(-1, 1)

    # If normalise_peak is True, then we divide the amplitudes by the maximum
    # value in each row. This ensures that the maximum value in each row is 1.
    # Else, we divide the amplitudes by the minimum value in each row. By doing
    # this, we ensure ensure the machine learning model prioritises the shape of
    # the peak rather than the amplitude of the peak. This is preferable when 
    # there is little to no noise in the signal.
    if normalise_peak:
        amplitudes = amplitudes / amplitudes.max(axis=1).reshape(-1, 1)

    # Add the normalized amplitudes to the dataframe and return the dataframe.
    df[amplitude_names] = amplitudes
    return df

def createWindows(
        df: pd.DataFrame,
        peak_window_radi: tuple[int, int],
        search_window_size: int,
        peak_threshold=0.1
    ) -> pd.DataFrame:
    """
    Create windows around the peaks in the data.
    
    :param df: a dataframe containing the amplitude column to create the windows
        around the peaks.
    :param peak_window_radi: The radius of the window around the peak. The first
        value is the number of samples before the peak and the second value is
        the number of samples after the peak.
    :param search_window_size: The size of the search window.
    :param peak_threshold: The minimum value that a peak can have to be
        considered a peak.
    
    :return: dataframe containing the windows around the peaks.
    """

    # We are going to create windows around the peaks. 

    # For the training data, we need to find the peaks in the data then link
    # each label to their own peak. We do this by creating a search window
    # around each label. We then find the first peak in the search window that
    # is greater than the peak threshold. We then link the label to the peak.
    # We then repeat this process for each label. However, there is a chance
    # that two labels are close enough to point to the same peak. To fix this
    # issue, we keep track of the peaks that have been assigned a label. If a
    # peak has already been assigned a label, then we skip it and move on to the
    # next peak within the search window. We repeat this process until we find a
    # peak that has not been assigned a label. If all the peaks within the
    # search window have been assigned a label, then we assign the last peak to
    # the label.

    # For the prediction data, we do not need to worry about the issue above as
    # we do not have labels. Therefore, we just find the first peak in the
    # search window that is greater than the peak threshold.
    
    # Get the indicies of the df
    indicies = df.index

    # Create a dataframe with the amplitude column duplicated window_size and
    # shifted by 1. This creates the search windows for the peaks on each row.
    df_windows = pd.concat([
        df['Amplitude'].shift(-i) for i in range(search_window_size)
    ], axis=1)

    # When we shift the amplitude column, we get a bunch of NaN values. We need
    # to fill these values with the last value in the amplitude column. We do
    # this by using the fill_value parameter. This ensures that no peaks are
    # detected beyond the amplitude column.
    df_windows = df_windows.ffill(axis=1)

    # Rename the 'Amplitude' columns to i where i is the amount the column was 
    # shifted by
    df_windows.columns = [str(i) for i in range(search_window_size)]

    # Create a column which will store the index of first peak in each window.
    # We will use this column to create the windows around the peaks. We will
    # then drop this column. The initial value of the column will be 0. If a
    # peak is found, then the value will be set to the index of the peak and the
    # window will be created around the peak. Else, the value will remain 0 and
    # the window will be created around the index or it can be dropped.
    window_peak_indicies = np.zeros(len(df_windows))

    # Find the first peak in the window that is larger than the peak threshold.
    # To do this, we duplicate the amplitude columns and shift them by -1 and 1.
    # We then find the indicies where the amplitude is greater than the
    # amplitude of the previous row.

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

    # Where the amplitude is greater than the amplitude of the previous row and
    # the amplitude of the next row, set the value to the amplitude. Else, set
    # the value to 0. This will give us a dataframe where the only values that
    # are not 0 are the peaks.
    highlight_peaks = np.where(
        (df_windows > df_windows_before) &
        (df_windows > df_windows_after) &
        (df_windows > peak_threshold),
        df_windows,
        df_windows * 0
    )

    # Get the absolute value of the highlight_peaks array in case there are
    # negative peaks.
    highlight_peaks_abs = np.abs(highlight_peaks)

    # Somne windows may not have a peak. Therefore, we need to find the indicies
    # of the rows where a peak was found. We do this by summing the values in
    # each row. If the sum is greater than 0, then a peak was found in the row.
    peak_row_indicies = np.where(highlight_peaks_abs.sum(axis=1) > 0)[0]

    # For each row that had a non-zero sum, get the index of the first value
    # that is bigger than 0. This is the index of the peak. However, this comes
    # with an issue for the labelled data when there are overlaps in spikes.

    # Take two labels, 1 and 2 and two peaks, A and B; both peaks occur 
    # after the labels but are close enough to be in the same search window.
    # Using the above method, the first peak, A, will be assigned to label 1
    # but will also be assigned to label 2. This is because the first peak, A,
    # is the first peak in the search window for both labels.

    # To fix this issue, we need to keep track of all the indicies of peak which
    # have been assigned a label. If a peak has already been assigned a label,
    # then we skip it and move on to the next peak within the search window. We
    # repeat this process until we find a peak that has not been assigned a
    # label. If all the peaks within the search window have been assigned a
    # label, then we assign the last peak to the label.

    # If the search windows was infintely large, then every peak would be
    # assigned a label as long as the peak exceeded the peak threshold.
    
    # Because both the training and prediction data are passed through this
    # function, we need to check if the label column exists. However, both the
    # training and prediction data have a label column. The difference is that
    # the prediction data has a label column filled with 0s. Therefore, we need
    # to get the indicies of the non-zero labels. If there are no non-zero
    # labels, then we are dealing with prediction data and therefore do not need
    # to worry about the issue above.
    indicies_of_labels = indicies[df['Label'] != 0]
    if len(indicies_of_labels) > 0:

        # Create an array to store the indicies of the peaks that have been
        # assigned a label
        indicies_with_labels = []

        # Loop through the indicies of the rows where a peak was found
        for peak_row_index in peak_row_indicies:

            # Get the indicies of the peaks in the row relative to the search
            # window. We then check if the index of the current window was
            # associated with a label. If it was, then we need to assign that
            # label to the peak. If it was not, then we assign the first peak to
            # the label. We then break out of the loop.
            peak_row = np.abs(highlight_peaks[peak_row_index])
            peak_col_indicies = np.where(peak_row > 0)[0]

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
        window_peak_indicies[peak_row_indicies] = np.argmax(
            highlight_peaks[peak_row_indicies] > 0,
            axis=1
        )

    # Perform the element wise addition of the indicies and the peak indicies.
    # This gets us the indicies of the peaks relative to the dataframe rather
    # than the search window.
    relative_peak_indicies = indicies + window_peak_indicies

    # Set the datatype of the relative_peak_indicies array to int so we can use
    # it to index the dataframe.
    relative_peak_indicies = relative_peak_indicies.astype(int)

    # Create a dataframe consisting of the windows around each index. We do this
    # by shifting the amplitude column by the peak_window_radi[0] and
    # peak_window_radi[1] values. We then concatenate the shifted amplitude
    # columns together. This gives us a dataframe where each row contains the
    # window around each index. We then rename the columns to 'Amplitudei' where
    # i is the amount the column was shifted by.
    df_windows = pd.concat([
        df['Amplitude'].shift(i, fill_value=0) for i in range(
            peak_window_radi[0],
            -peak_window_radi[1],
            -1
        )
    ], axis=1)

    df_windows.columns = [f'Amplitude{i}' for i in range(sum(peak_window_radi))]

    # df_windows contains windows for all the indicies in the dataframe. We only
    # want the windows around the peaks. Therefore, we get the rows where the
    # index is in the relative_peak_indicies array.
    df_windows = df_windows.loc[relative_peak_indicies, :]

    # Reset the indicies of the dataframe but keep the old indicies as a column
    # in the dataframe and name the column 'RelativePeakIndex'. This is the 
    # index position of where the peak was found relative to the original
    # signal. We need this column to return the locations of the peaks in the
    # prediction data.
    df_windows = df_windows.reset_index().rename(
        columns={'index': 'RelativePeakIndex'}
    )

    # df_windows does not contain the label, Time, or other columns. Therefore,
    # we need to add these columns to the dataframe. We do this by getting the
    # columns that are not in the df_windows dataframe and concatenating them
    # together.
    df_windows = pd.concat([
        df_windows,
        df[df.columns.difference(['Amplitude'])]
    ], axis=1)

    # Add the PeakIndex column to the dataframe df_windows and set the values to
    # the values in the window_peak_indicies array.
    df_windows['PeakIndex'] = window_peak_indicies

    return df_windows

def removeDuplicates(df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the duplicate windows from the dataframe.
    
    :param df_windows: dataframe containing the windows around the peaks
    
    :return: dataframe containing the windows around the peaks with no
        duplicates
    """

    # When removing duplicates, we only want to remove the duplicates where the
    # label is 0. This is because the windows where the label is not 0 are
    # important for training the model. Therefore, we split the dataframe into
    # two dataframes, one where the label is 0 and one where the label is not 0.
    df_null = df_windows[df_windows['Label'] == 0]
    df_not_null = df_windows[df_windows['Label'] != 0]

    # We remove the windows where the label is 0 and the PeakIndex is 0. This
    # is because is the PeakIndex is 0, then the window did not contain a peak.
    # Therefore, we do not want to keep the window. We then remove the
    # PeakIndex column as we no longer need it.
    df_null = df_null[df_null['PeakIndex'] != 0]
    df_null = df_null.drop(columns=['PeakIndex'])

    # We then remove any remaining duplicates specified by the
    # 'RelativePeakIndex' column. If there are two windows with the same
    # 'RelativePeakIndex' value, then the windows are based around the same
    # peak. Therefore, we only need to keep one of the windows. We then remove
    # the 'RelativePeakIndex' column as we no longer need it.
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

def removeOutliers(
        df: pd.DataFrame,
        peak_window_radi: tuple[int, int],
        threshold=1
    ) -> pd.DataFrame:
    """
    Remove the outliers from the dataframe
    
    :param df: dataframe containing the windows around the peaks
    :param threshold: threshold will be multiplied by the standard deviation of
        the 'Height' column. Any rows where the 'Height' column is greater than
        or less than threshold * standard deviation from the mean of the
        'Height' column will be removed.
    
    :return: dataframe
    """

    # Calculate the relative height of each window. This is the height of the
    # peak relative to the minimum value in the window.
    amplitude_names = df.filter(regex='Amplitude\d+').columns
    peak_amplitude_name = amplitude_names[peak_window_radi[0]]
    df['Height'] = df[peak_amplitude_name] - df[amplitude_names].min(axis=1)

    # Group the windows by the label column. This is because each label has a
    # different distribution of heights. Therefore, we need to remove the
    # outliers for each label separately.
    groups = df.groupby('Label')

    # For each group, remove the rows where the amplitude is greater than or
    # less than threshold * standard deviation from the mean of the 'Height'
    # column. We then add the group to a list of groups which we will concat
    # together at the end.
    list_of_groups = []
    for label, group in groups:
        # Calculate the distance from the mean of the 'Height' column
        distance_from_mean = np.abs(group['Height'] - group['Height'].mean())
        # Calculate the outlier range
        outlier_range = threshold * group['Height'].std()
        # Remove the outliers
        group = group[distance_from_mean < outlier_range]
        # Add the group to the list of groups
        list_of_groups.append(group)
    # Concat the groups together
    df = pd.concat(list_of_groups)

    # Drop the 'Height' column
    df = df.drop(columns=['Height'])

    return df

def unbiasData(
        df: pd.DataFrame,
        bias_coefficients=None
    ) -> pd.DataFrame:
    """
    Unbias the data by keeping the number of windows for each label the same.
    
    :param df: dataframe
    :param bias_coefficients: bias coefficients

    bias_coefficients is a list of coefficients for each label. The number of
    windows for each label will be multiplied by the corresponding coefficient.
    For example, if the bias_coefficients are [1, 2, 3, 4, 5, 6], then the
    number of windows for label 0 will be multiplied by 1, the number of windows
    for label 1 will be multiplied by 2, the number of windows for label 2 will
    be multiplied by 3, and so on. This cause bias towards the label with the
    highest coefficient. If bias_coefficients is None, then the number of
    windows for each label will be the same.
    
    :return: dataframe
    """

    # If bias_coefficients is None, then we set the bias_coefficients to [1, 1,
    # 1, 1, 1, 1]. This ensures that the number of windows for each label is the
    # same.
    if bias_coefficients is None:
        bias_coefficients = [1, 1, 1, 1, 1, 1]

    # To unbias the data, we need to ensure that all non-null labels (i.e., 
    # labels that are not 0) have the same number of windows. This ensures there
    # is no bias towards a particular label. However, we also want to keep the
    # number of windows for the null label (i.e., label 0) the same as the
    # total number of windows for the non-null labels. This ensures that the
    # model does not over predict the non-null labels.

    # Before either of these can be done, we need to find the minimum number of
    # windows for the non-null labels. This will be used as the number of
    # windows for the non-null labels and the null label.

    # We first get all of the one-hot encoded label columns and sum the values
    # in each column. This will give us the number of windows for each label. We
    # then find the minimum number of windows for the non-null labels.
    label_names = df.filter(regex='Label\d+').columns
    label_sums = df[label_names].sum()
    min_sum = int(label_sums.min())

    # To unbias the data, we group the rows by the label column. We then
    # randomly select min_sum  number of rows from each group. This ensures that
    # the number of windows for each label is the same.
    groups = df.groupby('Label')
    list_of_groups = []
    for label, group in groups:
        # Incase we want to bias the data towards a particular label, we
        # randomly select min_sum * bias_coefficients[label] number of rows from
        # each group causing bias towards the label with the highest
        # coefficient.
        num_reps = int(bias_coefficients[label])
        for rep in range(num_reps):
            # Randomly select min_sum number of rows from each group
            samples = group.sample(n=min_sum)
            # Add the samples to the list of groups
            list_of_groups.append(samples)

    # Concat the groups together
    df = pd.concat(list_of_groups)

    # Shuffle the dataframe to prevent the model from overfitting to a
    # particular label
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def postProcessData(
        df: pd.DataFrame,
        predictions: npt.NDArray,
        filepath: str
    ) -> None:
    """
    Post process the predictions and save them to the given filepath.

    :param df: prediction dataframe
    :param predictions: predictions made by the model
    :param filepath: filepath to save the predictions

    
    :return: None
    """
    # For some models, the predictions are a 2D array. For other models, the
    # predictions are a 1D array. If the predictions are a 2D array, then we
    # need to find the class with the highest probability. If the predictions
    # are a 1D array, then we can just use the predictions as the labels.
    if predictions.shape[0] != predictions.size:
        # Find the class with the highest probability
        prediction_labels = predictions.argmax(axis=1)
    else:
        prediction_labels = predictions

    # We can now add the predictions to the dataframe to the label column.
    df['Label'] = prediction_labels

    # Filter out the windows that are labelled as 0 as these are not considered
    # peaks.
    df = df[df['Label'] != 0]

    # Get the relative peak indicies and the labels. We minus 25 from the
    # relative peak indicies because the windows are created around the peak
    # but the labels in D1 are at the start of the peak. Therefore, we need to
    # subtract 25 from the relative peak indicies to get the index of the start
    # of the peak. We chose 25 because on average, the label, in D1, is 25
    # samples away from the peak.
    prediction_indicies = df['RelativePeakIndex'].values - 25
    prediction_labels = df['Label'].values

    # Save the predictions
    savePredictions(filepath, prediction_labels, prediction_indicies)

def savePredictions(
        filepath: str,
        prediction_labels: npt.NDArray,
        prediction_indicies: npt.NDArray
    ) -> None:
    """
    Save the predictions to the given filepath.

    :param filepath: filepath to save the predictions
    :param predictions_labels: predictions labels
    :param predictions_indicies: predictions indicies
    
    :return: None
    """

    # save the predictions as a .mat file
    sio.savemat(filepath, {
        'Class': prediction_labels,
        'Index': prediction_indicies
    })

def plotWindows(
        df: pd.DataFrame,
        window_radi: tuple[int, int],
        title: str
    ) -> None:
    """
    Plot the windows around the peaks.
    
    :param df: dataframe containing the windows around the peaks
    :param window_radi: The radius of the window around the peak. The first
        value is the number of samples before the peak and the second value is
        the number of samples after the peak.
    :param title: title of the plot

    We create a subplot for each label and plot the first 1000 windows for each
    label on the same graph with a small alpha value to make the graph less
    cluttered. This also highlights the average shape of each class of spike.
    
    :return: None
    """

    # Create a figure with 2 rows and 3 columns
    fig, ax = plt.subplots(2, 3)

    # Set the title of the figure
    fig.suptitle(title)

    # group the rows by the label column
    grouped = df.groupby('Label')

    # plot the first 1000 windows for each label on the same graph
    for i, (label, group) in enumerate(grouped):
        num_plots = len(group) if len(group) < 1000 else 1000

        # get the values in the amplitude columns
        amplitudes = group.filter(regex='Amplitude\d+').values
        plt.subplot(2, 3, i + 1)

        # plot the amplitudes as the color grey with an alpha of 0.5
        plt.plot(
            np.arange(-window_radi[0], window_radi[1]),
            amplitudes[:num_plots, :].T,
            color='grey',
            alpha=0.1
        )
        plt.xlabel('Window Index')
        plt.ylabel('Amplitude')
        plt.title(f'Class {label}')

    plt.tight_layout()
    
    # show the plot
    plt.show()

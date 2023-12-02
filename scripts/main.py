import scipy.io as sio
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow

import ai


def main():
    """
    Main function
    """

    batch_size = 100
    window_size = 100
    epochs = 10
    cutoff_freq = 1000
    sampling_freq = 25000
    
    d, index, label = loadData()

    # create a dataframe from the data
    df = createDataFrame(d, index, label)

    # normalize the data
    df_norm = normalizeData(df)

    # filter the data
    df_filtered = ai.lowPassFilter(df_norm, cutoff_freq, sampling_freq)

    # Split the data into windows
    df_windows = ai.splitIntoWindows(df_filtered, window_size)

    # # Split the data into training and testing sets
    df_train, df_test = ai.splitData(df_windows, 0.8)

    # Train the model
    model = ai.train(df_train, window_size, batch_size, epochs, len(df_train['Label'].unique()))

    # Test the model
    ai.test(model, df_test, window_size)

    # plot_data(index, label, df_norm, df_filtered)

def loadData():
    """
    Load the data
    
    :return: d, index, label
    """

    # Check if there is a command line argument
    if len(sys.argv) != 2:
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
    return d,index,label

def normalizeData(df):
    """
    Normalize the data
    
    :param df: dataframe
    
    :return: dataframe
    """
    # normalize the amplitude column so that the values are between -1 and 1 by
    # dividing by the maximum value
    df['Amplitude'] = df['Amplitude'] / df['Amplitude'].max()

    return df

def createDataFrame(d, index, label):
    """
    Create a dataframe from the data

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

    # convert the label column to a one-hot encoded vector
    labels =  tensorflow.keras.utils.to_categorical(df['Label'])

    # add the one-hot encoded vector to the dataframe
    for i in range(len(labels[0])):
        df['Label' + str(i)] = labels[:, i]

    # return the dataframe
    return df

def plot_data(index, label, df_norm, df_filtered, num_samples_plot=5000):
    """
    Plot the raw and filtered data
    
    :param index: the locations in the recording of the start of each spike
    :param label: the class of each spike (1, 2, 3, 4, or 5), i.e. the type of neuron that fired it
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


if __name__ == '__main__':
    main()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import preprocessing as pp
import machineLearning as ml


def main():
    """
    Main function
    """

    batch_size = 100
    window_size = 100
    epochs = 10
    cutoff_freq = 1000
    sampling_freq = 25000
    
    d, index, label = pp.loadData()

    # create a dataframe from the data
    df = pp.createDataFrame(d, index, label)

    # normalize the data
    df_norm = pp.normalizeAmplitudes(df)

    # filter the data
    df_filtered = pp.lowPassFilter(df_norm, cutoff_freq, sampling_freq)

    # Split the data into windows
    df_windows = pp.createWindows(df_filtered, window_size)

    # # Split the data into training and testing sets
    df_train, df_test = pp.getTrainAndTestData(df_windows, 0.8)


    numOutputs = len(df_train['Label'].unique())
    # Create the model
    model = ml.NeuralNetwork(window_size, numOutputs)
    # Train the model
    model.train(df_train, batch_size, epochs)

    # Test the model
    model.test(df_test)

    # plot_data(index, label, df_norm, df_filtered)



if __name__ == '__main__':
    main()

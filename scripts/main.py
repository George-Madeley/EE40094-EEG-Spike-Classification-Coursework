import machineLearning as ml
import preprocessing as pp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """
    Main function
    """

    # Create a list of the peak threshold values to use for each dataset
    peak_thresholds = [0.07, 0.13, 0.19, 0.31, 0.44]

    # Create a list for the low- and high-pass cutoff frequencies
    low_cutoff_freqs = [1000, 1000, 1000, 1000, 1000]
    high_cutoff_freqs = [100, 100, 100, 100, 100]

    # Create a list of the SNR values to use for each dataset
    noisePowers = [20, 40, 60, 80, 100]

    # Create a list of the data files to use for each dataset
    data_files = [
        './data/D2.mat',
        './data/D3.mat',
        './data/D4.mat', 
        './data/D5.mat', 
        './data/D6.mat'
    ]
    for i in range(5):
        peak_threshold = peak_thresholds[i]
        low_cutoff_freq = low_cutoff_freqs[i]
        high_cutoff_freq = high_cutoff_freqs[i]
        noisePower = noisePowers[i]
        filepath = data_files[i]
        run(filepath, noisePower, peak_threshold, low_cutoff_freq, high_cutoff_freq)
        print("\n\n")

def run(filepath, noisePower, peak_threshold, low_cutoff_freq, high_cutoff_freq):
    batch_size = 100
    peak_window_radius = 20
    search_window_size = 100
    epochs = 100
    sampling_freq = 25000
    training_partition = 1

    d, index, label = pp.loadTrainingData()
    # Preprocess the data
    df = pp.preprocessTrainingData(
        d,
        index,
        label,
        low_cutoff_freq,
        high_cutoff_freq,
        sampling_freq,
        peak_window_radius,
        search_window_size,
        noisePower
    )
    df_train, df_test = pp.getTrainAndTestData(df, training_partition)


    # Get the number of possible outputs
    numOutputs = len(df['Label'].unique())
    
    # Create the model
    model = ml.NeuralNetwork(peak_window_radius * 2, numOutputs)

    # Train the model
    model.train(df_train, batch_size, epochs)

    # If the training partition is less than 1, then the data was split into
    # training and testing sets. Therefore, we can test the model.
    if training_partition < 1:
        # Test the model
        return model.test(df_test)

    # Get filename
    filename = os.path.basename(filepath)
    # Predict the labels of the data in D2.mat, D3.mat, D4.mat, D5.mat, and
    # D6.mat
    print(f'Predicting {filename}...')
    # Load the data
    d = pp.loadPredictionData(filepath)

    # Preprocess the data
    df_prediction = pp.preprocessPredictionData(
        d,
        low_cutoff_freq,
        high_cutoff_freq,
        sampling_freq,
        peak_window_radius,
        search_window_size,
    )
    
    # Print the number of rows in the dataframe
    print(f'Number of rows in {filename}.mat: {len(df_prediction)}')

    # Make the predictions
    predictions, prediction_indicies = model.predict(df_prediction, title=filename)

    # calculate the number of predictions
    num_predictions = len(predictions)
    print(f'Number of predictions for {filename}.mat: {num_predictions}')

    # Save the predictions
    filepath = os.path.join('results', f'{filename}')
    pp.savePredictions(filepath, predictions, prediction_indicies)



if __name__ == '__main__':
    main()

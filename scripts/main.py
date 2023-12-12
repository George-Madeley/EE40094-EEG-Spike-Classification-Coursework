from models.neuralNetwork import NeuralNetwork
import preprocessing as pp
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """
    Main function
    """
    d, index, label = pp.loadTrainingData()

    run(d, index, label, training_partition=0.8)


def run(d, index, label):
    """
    Test the model

    :param d: data
    :param index: index
    :param label: label
    """

    # Get the keyword arguments
    batch_size = 100
    window_size = 100
    epochs = 10
    cutoff_freq = 1000
    sampling_freq = 25000
    prediction = False
    training_partition = 1

    # Preprocess the data
    df = pp.preprocessTrainingData(
        d,
        index,
        label,
        cutoff_freq,
        sampling_freq,
        window_size
    )

    # Split the data into training and testing sets
    df_train, df_test = pp.getTrainAndTestData(df, training_partition)

    # Get the number of possible outputs
    numOutputs = len(df['Label'].unique())

    # Create the model
    model = NeuralNetwork(window_size, numOutputs)

    # Train the model
    model.train(df, batch_size, epochs)

    # If the training partition is less than 1, then the data was split into
    # training and testing sets. Therefore, we can test the model.
    if training_partition < 1:
        # Test the model
        model.test(df_test)

    # Predict the labels of the data in D2.mat, D3.mat, D4.mat, D5.mat, and
    # D6.mat
    if prediction:
        for i in range(2, 7):
            print(f'Predicting D{i}...')
            # Load the data
            filepath = os.path.join('data', f'D{i}.mat')
            d = pp.loadPredictionData(filepath)

            # Preprocess the data
            df_prediction = pp.preprocessPredictionData(
                d, cutoff_freq, sampling_freq, window_size
            )

            # Make the predictions
            predictions = model.predict(df_prediction)

            # Save the predictions
            filepath = os.path.join('results', f'D{i}.mat')
            pp.savePredictions(filepath, predictions)

if __name__ == '__main__':
    main()

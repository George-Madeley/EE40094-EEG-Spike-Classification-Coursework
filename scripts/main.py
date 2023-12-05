import machineLearning as ml
import preprocessing as pp
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """
    Main function
    """

    heading = [
        'SNR',
        'filter_type',
        'Cutoff Frequency',
        'Batch Size',
        'Window Size',
        'Epochs',
        'Training',
        'Loss',
        'Accuracy',
        'Precision',
        'Recall']
    file_path = getResultsFileName('results.csv', heading, 'results')

    batch_size = 100
    window_size = 50
    epochs = 1000
    cutoff_freq = 1000
    sampling_freq = 25000
    SNR = 80
    filter_type = 'low'

    d, index, label = pp.loadTrainingData()

    loss, accuracy, precision, recall = run(
        d,
        index,
        label,
        training_partition=0.8,
        epochs=epochs,
        prediction=True,
    )

    result = [
        SNR,
        filter_type,
        cutoff_freq,
        batch_size,
        window_size,
        epochs,
        'Training',
        loss,
        accuracy,
        precision,
        recall
    ]
    writeResults(file_path, heading, result)


def run(d, index, label, **kwargs):
    """
    Test the model

    :param d: data
    :param index: index
    :param label: label
    :param kwargs: keyword arguments. Possible keyword arguments are:
        batch_size - the number of samples per gradient update
        window_size - the number of samples per window
        epochs - the number of epochs to train the model
        cutoff_freq - the cutoff frequency of the low pass filter
        sampling_freq - frequency at which the data was sampled
        training_partition - the amout of data to use for training the model.
        prediction - if True, then predict the labels of the data in D2.mat,
                        D3.mat, D4.mat, D5.mat, and D6.mat. The predictions
                        will be saved in the results directory.

    :return:
        loss - a list of the loss values for each epoch. Loss is the error
               of the model.
        accuracy - a list of the accuracy values for each epoch. Accuracy
                   is the percentage of correct predictions.
    """

    # Get the keyword arguments
    batch_size = kwargs.get('batch_size', 100)
    window_size = kwargs.get('window_size', 100)
    epochs = kwargs.get('epochs', 10)
    cutoff_freq = kwargs.get('cutoff_freq', 1000)
    sampling_freq = kwargs.get('sampling_freq', 25000)
    prediction = kwargs.get('prediction', False)
    training_partition = kwargs.get('training_partition', 1)

    # Preprocess the data
    df = pp.preprocessTrainingData(
        d,
        index,
        label,
        cutoff_freq,
        sampling_freq,
        window_size)

    # Split the data into training and testing sets
    df_train, df_test = pp.getTrainAndTestData(df, training_partition)

    # Get the number of possible outputs
    numOutputs = len(df['Label'].unique())

    # Create the model
    model = ml.NeuralNetwork(window_size, numOutputs)

    # Train the model
    model.train(df_train, batch_size, epochs)

    # Test the model
    loss, accuracy, precision, recall = test(training_partition, df_test, model)

    # Predict the labels of the data in D2.mat, D3.mat, D4.mat, D5.mat, and
    # D6.mat
    predict(window_size, cutoff_freq, sampling_freq, prediction, model)

    return loss, accuracy, precision, recall


def test(
        training_partition,
        df_test,
        model):
    # If the training partition is less than 1, then the data was split into
    # training and testing sets. Therefore, we can test the model.
    if training_partition < 1:
        # Test the model
        return model.test(df_test)


def predict(window_size, cutoff_freq, sampling_freq, prediction, model):

    # If prediction is True, then we want to predict the labels of the data in
    # D2.mat, D3.mat, D4.mat, D5.mat, and D6.mat. The predictions will be saved
    # in the results directory.
    if prediction:
        for i in range(2, 7):
            print(f'Predicting D{i}...')
            # Load the data
            filepath = os.path.join('data', f'D{i}.mat')
            d = pp.loadPredictionData(filepath)

            # Preprocess the data
            df_prediction = pp.preprocessPredictionData(
                d, cutoff_freq, sampling_freq, window_size)

            # Make the predictions
            predictions, prediction_indicies = model.predict(df_prediction)

            # calculate the number of predictions
            num_predictions = len(predictions)
            print(f'Number of predictions for D{i}.mat: {num_predictions}')

            # Save the predictions
            filepath = os.path.join('results', f'D{i}.mat')
            pp.savePredictions(filepath, predictions, prediction_indicies)


def getResultsFileName(name: str, header: list[str], *directories) -> str:
    """
    Get the results file name. Create the file and directories if they do not
    exist.

    :param name: name of the file
    :param header: header of the file
    :param directories: directories to find the file in.

    :return: file path
    """

    # For each directory in directories, create the directory if it does not
    # exist.
    directory_path = ""
    for directory in directories:
        directory_path = os.path.join(directory_path, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Create the file path
    file_path = os.path.join(directory_path, name)

    # Create the file and add the header if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    return file_path


def writeResults(file_path: str, headers: list[str], row: list) -> None:
    """
    Write the results to the file.

    :param file_path: file path
    :param headers: headers
    :param row: row

    :return: None
    """

    # Create a dictionary from the headers and row
    results = dict(zip(headers, row))

    # try and write to the file. If the file does not exist, create it.
    try:
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(results)
    except FileNotFoundError:
        # If the file does not exist, create it and write the results to it. To
        # create the file, we need to get the file name, the directroies the
        # file is in, and the headers.
        # To get the directories, we need to split the file path into a list of
        # directories.
        directories = os.path.split(file_path)[0].split(os.path.sep)
        # Get the file name
        file_name = os.path.split(file_path)[1]
        file_path = getResultsFileName(file_path, headers)
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(results)


if __name__ == '__main__':
    main()

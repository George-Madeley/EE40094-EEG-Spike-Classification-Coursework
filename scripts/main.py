import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import preprocessing as pp
import machineLearning as ml


def main():
    """
    Main function
    """

    heading = ['SNR', 'filter_type', 'Cutoff Frequency', 'Layer Type', 'Batch Size', 'Window Size', 'Epochs', 'Epoch No.', 'Training', 'Accuracy', 'Loss']
    file_path = getResultsFileName('results.csv', heading, 'results')

    batch_size = 100
    window_size = 100
    epochs = 1
    cutoff_freq = 1000
    sampling_freq = 25000
    SNR = 80
    filter_type = 'low'
    layer_type = 'Dense'

    
    d, index, label = pp.loadTrainingData()

    losses, accuracies = test(d, index, label, batch_size, window_size, epochs, cutoff_freq, sampling_freq, prediction=True)

    for i in range(epochs):
        result = [SNR, filter_type, cutoff_freq, layer_type, batch_size, window_size, epochs, i, 'Training', losses[i], accuracies[i]]
        writeResults(file_path, heading, result)
    result = [SNR, filter_type, cutoff_freq, layer_type, batch_size, window_size, epochs, -1, 'Testing', losses[-1], accuracies[-1]]
    writeResults(file_path, heading, result)
    

def test(d, index, label, batch_size=100, window_size=100, epochs=10, cutoff_freq=1000, sampling_freq=25000, prediction=False):
    """
    Test the model

    :param batch_size: batch size
    :param window_size: window size
    :param epochs: epochs
    :param cutoff_freq: cutoff frequency
    :param sampling_freq: sampling frequency
    :param d: data
    :param index: index
    :param label: label
    :param prediction: prediction

    :return:
        loss - a list of the loss values for each epoch. Loss is the error
               of the model.
        accuracy - a list of the accuracy values for each epoch. Accuracy
                   is the percentage of correct predictions.
    """

    # Preprocess the data
    df = pp.preprocessTrainingData(d, index, label, cutoff_freq, sampling_freq, window_size)

    # the amout of data to use for training the model. The rest of the data will
    # be used for testing the model.
    training_partition = 1

    # Split the data into training and testing sets
    df_train, df_test = pp.getTrainAndTestData(df, training_partition)

    numOutputs = len(df_train['Label'].unique())
    # Create the model
    model = ml.NeuralNetwork(window_size, numOutputs)
    # Train the model
    losses, accuracies = model.train(df_train, batch_size, epochs)

    # If the training partition is less than 1, then the data was split into
    # training and testing sets. Therefore, we can test the model.
    if training_partition < 1:
        # Test the model
        loss, accuracy = model.test(df_test)

        losses.append(loss)
        accuracies.append(accuracy)

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
            df_prediction = pp.preprocessPredictionData(d, cutoff_freq, sampling_freq, window_size)
            
            # Make the predictions
            predictions, prediction_indicies = model.predict(df_prediction)

            # Save the predictions
            filepath = os.path.join('results', f'D{i}.mat')
            pp.savePredictions(filepath, predictions, prediction_indicies)

    return losses, accuracies

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

import machineLearning as ml
from preprocessing import preprocessData, savePredictions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    """
    Main function
    """
    # Create a list of the data files to use for each dataset
    filepaths = [
        './data/D2.mat',
        './data/D3.mat',
        './data/D4.mat', 
        './data/D5.mat', 
        './data/D6.mat'
    ]
    for filepath in filepaths:
        run(filepath)
        print("\n\n")

def run(filepath):
    batch_size = 100
    peak_window_radius = 30
    search_window_size = 100
    epochs = 100
    sampling_freq = 25000

    df_train, df_predi = preprocessData(
        filepath,
        sampling_freq=sampling_freq,
        peak_window_radius=peak_window_radius,
        search_window_size=search_window_size,
    )


    # Get the number of possible outputs
    numOutputs = len(df_train['Label'].unique())
    
    # Create the model
    model = ml.NeuralNetwork(peak_window_radius * 2, numOutputs)

    # Train the model
    model.train(df_train, batch_size, epochs)

    # Get filename
    filename = os.path.basename(filepath)
    # Predict the labels of the data in D2.mat, D3.mat, D4.mat, D5.mat, and
    # D6.mat
    print(f'Predicting {filename}...')
    
    # Print the number of rows in the dataframe
    print(f'Number of rows in {filename}.mat: {len(df_predi)}')

    # Make the predictions
    predictions, prediction_indicies = model.predict(df_predi, title=filename)

    # calculate the number of predictions
    num_predictions = len(predictions)
    print(f'Number of predictions for {filename}.mat: {num_predictions}')

    # Save the predictions
    filepath = os.path.join('results', f'{filename}')
    savePredictions(filepath, predictions, prediction_indicies)

if __name__ == '__main__':
    main()

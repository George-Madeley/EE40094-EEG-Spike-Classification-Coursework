from models.neuralNetwork import NeuralNetwork
from models.kNearestNeighbor import KNearestNeighbor
from preprocessing import preprocessData, postProcessData
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
    peak_window_radi = (30, 30)
    search_window_size = 100
    epochs = 100
    sampling_freq = 25000
    regex = 'Amplitude\d+'
    PCA = regex == 'PC\d+'
    K = 3

    df_train, df_predi = preprocessData(
        filepath,
        sampling_freq=sampling_freq,
        peak_window_radi=peak_window_radi,
        search_window_size=search_window_size,
        PCA=PCA,
    )

    # Get the number of possible outputs
    numOutputs = len(df_train['Label'].unique())
    # get the number of possible inputs
    numInputs = len(df_train.filter(regex=regex).columns)
    
    # Create the model
    model = KNearestNeighbor(K)

    # Train the model
    model.train(df_train, regex)

    # Get filename
    filename = os.path.basename(filepath)
    print(f'Predicting {filename}...')
    
    # Print the number of rows in the dataframe
    print(f'Number of rows in {filename}.mat: {len(df_predi)}')

    # Make the predictions
    predictions = model.predict(df_predi, regex)
    
    filepath = os.path.join('results', f'{filename}')

    postProcessData(df_predi, predictions, filepath, peak_window_radi)



if __name__ == '__main__':
    main()

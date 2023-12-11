import machineLearning as ml
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
    # get the number of possible inputs
    numInputs = len(df_train.filter(regex='PCA\d+').columns)
    
    # Create the model
    model = ml.NeuralNetwork(numInputs, numOutputs)

    # Train the model
    model.train(df_train, batch_size, epochs)

    # Get filename
    filename = os.path.basename(filepath)
    print(f'Predicting {filename}...')
    
    # Print the number of rows in the dataframe
    print(f'Number of rows in {filename}.mat: {len(df_predi)}')

    # Make the predictions
    predictions = model.predict(df_predi)
    
    filepath = os.path.join('results', f'{filename}')

    postProcessData(df_predi, predictions, filepath)



if __name__ == '__main__':
    main()

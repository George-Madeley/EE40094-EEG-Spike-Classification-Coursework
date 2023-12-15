from models.kNearestNeighbor import KNearestNeighbor
from dataProcessing import preprocessData, postProcessData
import os

def main():
    """
    Creates and trains a model to classify types of signals spikes in a series
    of data files. The model is then used to predict the types of spikes in
    another series of data files.

    The model is trained by first preprocessing the data. The preprocessing
    involves finding the peaks in the signal and extracting the windows around
    the peaks. The windows are then normalised and passed to the model to be
    trained.

    The model is then used to predict the types of spikes in another series of
    data files. The data files are preprocessed in the same way as the training
    data. The model then predicts the types of spikes in the data files and
    saves the predictions to a file.
    """
    # Create a list of filepaths to the data files. These are the signal files
    # that we want to classify.
    filepaths = [
        './data/D2.mat',
        './data/D3.mat',
        './data/D4.mat', 
        './data/D5.mat', 
        './data/D6.mat'
    ]
    # Each signal file has a different peak threshold because they have
    # different noise levels. The peak threshold is the minimum value that a
    # peak can have to be considered a peak.
    peak_thresholds = [0.03, 0.06, 0.09, 0.18, 0.23]
    # Each signal file has a different normalisation method. The normalisation
    # method is used to normalise the peaks in the signal. If the value is
    # True, the peaks are normalised by the maximum value in the windows, else
    # the peaks are normalised by the minimum value in the windows.
    normalise_peaks = [True, True, True, False, False]

    # Run the run function for each file in the filepaths list.
    for i, filepath in enumerate(filepaths):
        peak_threshold = peak_thresholds[i]
        normalise_peak = normalise_peaks[i]
        run(filepath, peak_threshold, normalise_peak)
        print("\n\n")

def run(filepath: str, peak_threshold: float, normalise_peak: bool) -> None:
    """
    Create the model, train it and make predictions.
    
    :param filepath: The path to the data file.
    :param peak_threshold: The minimum value that a peak can have to be
        considered a peak.
    :param normalise_peak: If True, the peaks are normalised by the maximum
        value in the windows, else the peaks are normalised by the minimum value
        in the windows.

    :return: None
    """
    # Set the parameters for the model.
    # peak_window_radi is the radius of the window around the peak. The first
    # value is the number of samples before the peak and the second value is
    # the number of samples after the peak.
    peak_window_radi = (30, 60)
    # To find the peaks in the signal, a search window is used. For each sample
    # in the signal, a window of size search_window_size is created starting
    # from the sample. If a peak is found in the window, the location of the
    # peak is saved and the window is moved forward by 1.
    # Search window size is the size of the search window.
    search_window_size = 100
    # K is the number of nearest neighbors to use in the K nearest neighbors
    # algorithm.
    K = 3

    # Preprocess the data.
    df_train, df_predi = preprocessData(
        filepath,
        peak_window_radi=peak_window_radi,
        search_window_size=search_window_size,
        peak_threshold=peak_threshold,
        normalise_peak=normalise_peak
    )
    
    # Create the K nearest neighbors model
    model = KNearestNeighbor(K)

    # Train the model on the training data
    model.train(df_train)


    # Make the predictions on the prediction data
    predictions = model.predict(df_predi)
    
    # Create the save filepath for the predictions
    filename = os.path.basename(filepath)
    filepath = os.path.join('results', f'{filename}')

    # Postprocess the predictions and save them to a file
    postProcessData(df_predi, predictions, filepath, peak_window_radi)


# Run the main function when this file is run.
if __name__ == '__main__':
    main()

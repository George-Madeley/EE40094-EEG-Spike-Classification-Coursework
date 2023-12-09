from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv1D, Flatten
from keras.metrics import Precision, Recall
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import tensorflow as tf

from preprocessing import plotWindows

from IArtificialIntelligence import IArtificialIntelligence

class NeuralNetwork(IArtificialIntelligence):
    """
    Neural network class. Used to create, train and test a neural network model.
    """
    def __init__(self, numInputs, numOutputs):
        self.model = self.createModel(numInputs, numOutputs)

    def createModel(self, numInputs, numOutputs):
        """
        Create a neural network model
        
        :param numInputs: number of inputs
        :param numOutputs: number of outputs
        
        :return: model
        """

        num_hidden_neurons = 100
        kernel_size = 3

        model = Sequential()
        model.add(InputLayer(input_shape=(numInputs, 1)))
        model.add(Conv1D(num_hidden_neurons, kernel_size=kernel_size, activation='relu'))
        model.add(Flatten())
        model.add(Dense(numOutputs, activation='softmax'))

        opt = Adam(learning_rate=0.001)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])

        return model
    
    def train(self, df_train, batch_size, epochs):
        """
        Train the model
        
        :param train_df: training dataframe
        :param batch_size: batch size
        :param epochs: epochs
        
        :return: None
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_train.filter(regex='Amplitude\d+').columns
        amplitudes = df_train[amplitude_names].values

        # Add a dimension to the amplitudes array for the convolutional layer
        amplitudes = np.expand_dims(amplitudes, axis=2)
        # turn amplitudes into a tensor
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_train.filter(regex='Label\d+').columns
        labels = df_train[label_names].values

        # turn labels into a tensor
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Add early stopping
        early_stopping = EarlyStopping(monitor='val_precision', patience=20, mode='max')

        # Train the model
        self.model.fit(amplitudes, labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, callbacks=[early_stopping])

    def test(self, df_test):
        """
        Test the model

        :param test_df: test dataframe

        :return:
            loss - the error of the model.
            accuracy - the percentage of correct predictions.
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_test.filter(regex='Amplitude\d+').columns
        amplitudes = df_test[amplitude_names].values

        # Add a dimension to the amplitudes array for the convolutional layer
        amplitudes = np.expand_dims(amplitudes, axis=2)

        # Turn amplitudes into a tensor
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_test.filter(regex='Label\d+').columns
        labels = df_test[label_names].values

        # Turn labels into a tensor
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Evaluate the model
        history = self.model.evaluate(amplitudes, labels, verbose=1)

        loss = history[0]
        accuracy = history[1]
        precision = history[2]
        recall = history[3]

        return loss, accuracy, precision, recall
    
    def predict(self, df, peak_threshold=0, title=''):
        """
        Predict the class of each window

        :param df: dataframe
        :param peak_threshold: peak threshold

        :return: predictions, predictions_indicies
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df.filter(regex='Amplitude\d+').columns
        amplitudes = df[amplitude_names].values

        # Add a dimension to the amplitudes array for the convolutional layer
        amplitudes = np.expand_dims(amplitudes, axis=2)

        # turn amplitudes into a tensor
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Predict the class of each window
        predictions = self.model.predict(amplitudes)

        # Find the class with the highest probability
        prediction_labels = predictions.argmax(axis=1)

        # Create a dataframe of the predictions
        df_predictions = df.copy()
        df_predictions['Label'] = prediction_labels

        # Plot the windows
        plotWindows(df_predictions, 50, f'Predictions for {title}')

        # Filter out the windows that are labelled as 0
        df_predictions = df_predictions[df_predictions['Label'] != 0]

        # Get the first 3000 windows
        df_predictions = df_predictions[:3000]

        # Get the indicies of the predictions
        prediction_indicies = df_predictions.index.values
        prediction_labels = df_predictions['Label'].values

        return prediction_labels, prediction_indicies
    
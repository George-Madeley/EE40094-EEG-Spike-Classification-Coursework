from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv1D, Flatten
from keras.metrics import Precision, Recall
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter

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
        self.model.fit(amplitudes, labels, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.2, callbacks=[early_stopping])

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
    
    def predict(self, df, probability_threshold=0.1):
        """
        Predict the class of each window

        :param df: dataframe
        :param probability_threshold: The threshold the predictions must exceed
                                        to be considered a positive prediction.

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

        # Perform a max convolution on the amplitude0 column with a kernel size
        # equal to the window size.
        amplitude0 = df['Amplitude0'].values
        amplitude0 = maximum_filter(amplitude0, size=40)

        # num_samples = 5000

        # # Create a subplot with 2 rows and 1 column
        # fig, ax = plt.subplots(3, 1)

        # # For the first subplot, plot the ampltiudes of the first 5000 samples
        # plt.subplot(3, 1, 1)
        # plt.plot(amplitude0[:num_samples])
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Amplitude of the First 5000 Samples')

        # # Plot a line graph with window number on the x-axis and the probability
        # # of each class on the y-axis for the first 5000 samples. Plot each 
        # # class on the same graph.
        # plt.subplot(3, 1, 2)
        # for i in range(len(predictions[0])):
        #     plt.plot(predictions[:num_samples, i], label='Class ' + str(i))
        # plt.xlabel('Window Number')
        # plt.ylabel('Probability')
        # plt.title('Probability of Each Class for the First 5000 Samples')
        # plt.legend()

        # # perform element-wise multiplication of the probabilities and the
        # # amplitude0
        # product_probabilities = predictions * amplitude0[:, np.newaxis]

        # plt.subplot(3, 1, 3)
        # for i in range(len(predictions[0])):
        #     plt.plot(product_probabilities[:num_samples, i], label='Class ' + str(i))
        # plt.xlabel('Window Number')
        # plt.ylabel('Probability')
        # plt.title('Probability of Each Class for the First 5000 Samples')
        # plt.legend()

        # plt.show()

        # For each label in the predictions, find the indicies of the
        # probabilities that are greater than the probability before and after
        # it. This will give us the indicies of the peaks in the predictions.
        product_probabilities[:, 0] = 0
        for i in range(1, product_probabilities.shape[1]):
            # calculate the before and after probabilities
            before = np.roll(product_probabilities[:, i], 1)
            after = np.roll(product_probabilities[:, i], -1)

            # Find the indicies of the peaks
            indicies = np.where(
                (product_probabilities[:, i] > before) &
                (product_probabilities[:, i] > after) &
                (product_probabilities[:, i] >= probability_threshold)
            )[0]

            # Get the probabilities at the peaks
            probabilities = product_probabilities[indicies, i]

            # Set all the probabilities to 0
            product_probabilities[:, i] = 0

            # Set the probabilities at the peaks to the probabilities
            product_probabilities[indicies, i] = probabilities

        # Find the class with the highest probability
        product_probabilities = product_probabilities.argmax(axis=1)

        # Get the indicies where the predictions are not 0
        predictions_indicies = np.where(product_probabilities != 0)[0]

        # Get the predictions that are not 0
        predictions = product_probabilities[predictions_indicies]

        return predictions, predictions_indicies
    
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import tensorflow as tf
import pandas as pd

from models.IArtificialIntelligence import IArtificialIntelligence

class NeuralNetwork(IArtificialIntelligence):
    """
    Neural network class. Used to create, train and test a neural network model.
    """
    def __init__(self, numInputs: int, numOutputs: int) -> None:
        """
        Create a neural network model

        :param numInputs: number of input neurons
        :param numOutputs: number of output neurons

        :return: None
        """
        
        # Create the model and save it
        self.model = self.createModel(numInputs, numOutputs)

    def createModel(self, numInputs: int, numOutputs: int) -> object:
        """
        Create a neural network model
        
        :param numInputs: number of inputs neurons
        :param numOutputs: number of outputs neurons
        
        :return: model
        """

        # Create the model with the given number of inputs and outputs and 
        # two hidden layers with 60 and 30 neurons respectively and a sigmoid
        # activation function. The output layer has a softmax activation
        # function. We use the softmax activation function as we want the
        # output to be a probability distribution over the classes.
        model = Sequential()
        model.add(InputLayer(input_shape=(numInputs,)))
        model.add(Dense(60, activation='sigmoid'))
        model.add(Dense(30, activation='sigmoid'))
        model.add(Dense(numOutputs, activation='softmax'))

        # Create the optimizer
        opt = Adam(learning_rate=0.001)

        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        # return the model
        return model
    
    def train(
            self,
            df_train: pd.DataFrame,
            batch_size: int,
            epochs: int
        ) -> None:
        """
        Train the model
        
        :param train_df: training dataframe
        :param batch_size: the number of samples per gradient update
        :param epochs: the number of iterations to train the model
        
        :return: None
        """

        # Get the the columns that start with 'Amplitude' and suffix with a
        # number
        amplitude_names = df_train.filter(regex='Amplitude\d+').columns
        amplitudes = df_train[amplitude_names].values

        # turn amplitudes into a tensor as the model runs faster with tensors
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_train.filter(regex='Label\d+').columns
        labels = df_train[label_names].values

        # turn labels into a tensor as the model runs faster with tensors
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Add early stopping to stop when the loss begins to increase
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            mode='min',
            restore_best_weights=True
        )

        # Train the model
        self.model.fit(
            amplitudes,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

    def test(self, df_test: pd.DataFrame) -> tuple[float, float]:
        """
        Test the model

        :param test_df: test dataframe

        :return:
            loss - the error of the model.
            accuracy - the percentage of correct predictions.
        """

        # Get the the columns that start with 'Amplitude' and suffix with a
        # number
        amplitude_names = df_test.filter(regex='Amplitude\d+').columns
        amplitudes = df_test[amplitude_names].values

        # Turn amplitudes into a tensor as the model runs faster with tensors
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_test.filter(regex='Label\d+').columns
        labels = df_test[label_names].values

        # Turn labels into a tensor as the model runs faster with tensors
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Evaluate the model
        history = self.model.evaluate(amplitudes, labels, verbose=1)

        # Get the loss and accuracy
        loss = history[0]
        accuracy = history[1]

        return loss, accuracy
    
    def predict(self, df: pd.DataFrame) -> list[float]:
        """
        Predict the class of each window

        :param df: dataframe

        :return: predictions
        """

        # Get the the columns that start with 'Amplitude' and suffix with a
        # number
        amplitude_names = df.filter(regex='Amplitude\d+').columns
        amplitudes = df[amplitude_names].values

        # turn amplitudes into a tensor
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Predict the class of each window
        predictions = self.model.predict(amplitudes)

        # Return the predictions
        return predictions
    
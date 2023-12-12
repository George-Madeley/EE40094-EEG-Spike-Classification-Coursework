from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import tensorflow as tf

from models.IArtificialIntelligence import IArtificialIntelligence

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

        model = Sequential()
        model.add(InputLayer(input_shape=(numInputs,)))
        model.add(Dense(60, activation='sigmoid'))
        model.add(Dense(30, activation='sigmoid'))
        model.add(Dense(numOutputs, activation='softmax'))

        opt = Adam(learning_rate=0.001)

        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

        return model
    
    def train(self, df_train, batch_size, epochs, regex):
        """
        Train the model
        
        :param train_df: training dataframe
        :param batch_size: batch size
        :param epochs: epochs
        :param regex: regex to use to get the columns
        
        :return: None
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_train.filter(regex=regex).columns
        amplitudes = df_train[amplitude_names].values

        # turn amplitudes into a tensor
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_train.filter(regex='Label\d+').columns
        labels = df_train[label_names].values

        # turn labels into a tensor
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

    def test(self, df_test, regex):
        """
        Test the model

        :param test_df: test dataframe
        :param regex: regex to use to get the columns

        :return:
            loss - the error of the model.
            accuracy - the percentage of correct predictions.
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_test.filter(regex=regex).columns
        amplitudes = df_test[amplitude_names].values

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
    
    def predict(self, df, regex):
        """
        Predict the class of each window

        :param df: dataframe
        :param regex: regex to use to get the columns

        :return: predictions
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df.filter(regex=regex).columns
        amplitudes = df[amplitude_names].values

        # turn amplitudes into a tensor
        amplitudes = tf.convert_to_tensor(amplitudes, dtype=tf.float32)

        # Predict the class of each window
        predictions = self.model.predict(amplitudes)

        return predictions
    
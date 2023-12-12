from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.metrics import Precision, Recall

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

        # Create a sequential neural network model with 1 input layer, n hidden
        # layers and 1 output layer.
        model = Sequential()
        model.add(InputLayer(input_shape=(numInputs,)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        
        # Add the output layer. The activation function is softmax, which is
        # used for multi-class classification.
        model.add(Dense(numOutputs, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])

        return model
    
    def train(self, df_train, batch_size, epochs):
        """
        Train the model
        
        :param train_df: training dataframe
        :param batch_size: batch size
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_train.filter(regex='Amplitude\d+').columns
        amplitudes = df_train[amplitude_names].values

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_train.filter(regex='Label\d+').columns
        labels = df_train[label_names].values

        # Train the model
        self.model.fit(amplitudes, labels, batch_size=batch_size, epochs=epochs, verbose=1)

    def test(self, df_test):
        """
        Test the model

        :param test_df: test dataframe
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_test.filter(regex='Amplitude\d+').columns
        amplitudes = df_test[amplitude_names].values

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_test.filter(regex='Label\d+').columns
        labels = df_test[label_names].values

        # Evaluate the model
        self.model.evaluate(amplitudes, labels, verbose=1)
    
    def predict(self, df):
        """
        Predict the class of each window

        :param df: dataframe

        :return: predictions
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df.filter(regex='Amplitude\d+').columns
        amplitudes = df[amplitude_names].values

        # Predict the class of each window
        predictions = self.model.predict(amplitudes)

        return predictions
    
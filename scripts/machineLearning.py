from keras.models import Sequential
from keras.layers import Dense, InputLayer

from IArtificialIntelligence import IArtificialIntelligence

class NeuralNetwork(IArtificialIntelligence):
    """
    Neural network class. Used to create, train and test a neural network model.
    """
    def __init__(self, numInputs, numOutputs):
        self.model = self.createModel(numInputs, numOutputs)

    def createModel(self, numInputs, numOutputs, numHiddenLayers=None):
        """
        Create a neural network model
        
        :param numInputs: number of inputs
        :param numOutputs: number of outputs
        :param numHiddenLayers: A list of dictoinaries containing the number of
                                neurons in each hidden layer, the activation,
                                and the type of layer.
        
        :return: model
        """

        # Set the default number of hidden layers if none is given
        if numHiddenLayers is None:
            numHiddenLayers = [
                {'num_neurons': 100, 'activation': 'relu', 'type': 'dense'},
                {'num_neurons': 100, 'activation': 'relu', 'type': 'dense'},
            ]

        # Create a sequential neural network model with 1 input layer, n hidden
        # layers and 1 output layer.
        model = Sequential()
        model.add(InputLayer(input_shape=(numInputs,)))

        for layer_data in numHiddenLayers:
            if layer_data.get('type') == 'dense':
                model.add(Dense(layer_data.get('num_neurons', 100), activation=layer_data.get('activation', 'relu')))
            else:
                model.add(Dense(layer_data.get('num_neurons', 100), activation=layer_data.get('activation', 'relu')))
        
        # Add the output layer. The activation function is softmax, which is
        # used for multi-class classification.
        model.add(Dense(numOutputs, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
    
    def train(self, df_train, batch_size, epochs):
        """
        Train the model
        
        :param train_df: training dataframe
        :param batch_size: batch size
        :param epochs: epochs
        
        :return:
            loss - a list of the loss values for each epoch. Loss is the error
                   of the model.
            accuracy - a list of the accuracy values for each epoch. Accuracy
                       is the percentage of correct predictions.
        """

        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_train.filter(regex='Amplitude\d+').columns
        amplitudes = df_train[amplitude_names].values

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_train.filter(regex='Label\d+').columns
        labels = df_train[label_names].values

        # Train the model
        history = self.model.fit(amplitudes, labels, batch_size=batch_size, epochs=epochs, verbose=1)

        loss = history.history.get('loss',[])
        accuracy = history.history.get('accuracy',[])

        return loss, accuracy

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

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_test.filter(regex='Label\d+').columns
        labels = df_test[label_names].values

        # Evaluate the model
        loss, accuracy = self.model.evaluate(amplitudes, labels, verbose=1)

        print('Loss:', loss)
        print('Accuracy:', accuracy)

        return loss, accuracy
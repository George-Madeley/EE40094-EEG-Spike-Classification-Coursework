from abc import ABC, abstractmethod

class IArtificialIntelligence(ABC):
    @abstractmethod
    def train(self, data):
        """
        Train the model
        
        :param train_df: training dataframe
        :param batch_size: batch size
        :param epochs: number of training iterations
        """
        pass

    @abstractmethod
    def test(self, data):
        """
        Test the model
        
        :param test_df: test dataframe
        
        :return: loss, accuracy
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Predict the labels of the test data
        
        :param test_df: test dataframe
        
        :return: predictions
        """
        pass

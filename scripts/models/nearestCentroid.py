from sklearn.neighbors import NearestCentroid as NearestCentroidClassifier

import pandas as pd

from models.IArtificialIntelligence import IArtificialIntelligence

class NearestCentroid(IArtificialIntelligence):
    def __init__(self) -> None:
        """
        Create a nearest centroid model
        
        :return: None
        """
        # Create the model and save it
        self.model = self.createModel()

    def createModel(self) -> object:
        """
        Create a nearest centroid model
        
        :return: model
        """

        # Create the model
        model = NearestCentroidClassifier()

        return model

    def train(self, df_train: pd.DataFrame) -> None:
        """
        Train the model
        
        :param train_df: training dataframe

        :return: None
        """
        # Get the the columns that start with 'Amplitude' and suffix with a
        # number
        amplitude_names = df_train.filter(regex='Amplitude\d+').columns
        amplitudes = df_train[amplitude_names].values

        # Get the column that start with 'Label' and no suffix as the model does
        # not support one hot encoding.
        label_names = df_train.filter(regex='Label\d+').columns
        label_names = df_train.filter(regex='Label').columns.difference(
            label_names
        )
        labels = df_train[label_names].values.ravel()

        # Train the model
        self.model.fit(amplitudes, labels)

    def test(self, df_test: pd.DataFrame) -> float:
        """
        Test the model
        
        :param test_df: test dataframe
        
        :return: score
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_test.filter(regex='Amplitude\d+').columns
        amplitudes = df_test[amplitude_names].values

        # Get the column that start with 'Label' and no suffix as the model does
        # not support one hot encoding.
        label_names = df_test.filter(regex='Label\d+').columns
        label_names = df_test.filter(regex='Label').columns.difference(
            label_names
        )
        labels = df_test[label_names].values.ravel()

        # Test the model
        score = self.model.score(amplitudes, labels)

        # Return the score
        return score

    def predict(self, df_predictions: pd.DataFrame) -> list[float]:
        """
        Predict the labels of the test data
        
        :param test_df: test dataframe
        
        :return: predictions
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_predictions.filter(regex='Amplitude\d+').columns
        amplitudes = df_predictions[amplitude_names].values

        # Predict the labels
        predictions = self.model.predict(amplitudes)

        # Return the predictions
        return predictions

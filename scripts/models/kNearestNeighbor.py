from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

from models.IArtificialIntelligence import IArtificialIntelligence

class KNearestNeighbor(IArtificialIntelligence):
    def __init__(self, k: int) -> None:
        """
        Create a k nearest neighbor model
        
        :param k: number of neighbors
        
        :return: None
        """
        # Create the model and save it
        self.model = self.createModel(k)

    def createModel(self, k: int) -> object:
        """
        Create a k nearest neighbor model
        
        :param k: number of neighbors
        
        :return: model
        """

        # Create the model with the given number of neighbors and use the
        # distance to the neighbors to weight the votes.
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')

        return model

    def train(self, df_train: pd.DataFrame) -> None:
        """
        Train the model
        
        :param train_df: training dataframe
        """
        # Get the the columns that start with 'Amplitude' and suffix with a
        # number
        amplitude_names = df_train.filter(regex='Amplitude\d+').columns
        amplitudes = df_train[amplitude_names].values

        # Get the column that start with 'Label' and no suffix
        label_names = df_train.filter(regex='Label\d+').columns
        label_names = df_train.filter(regex='Label').columns.difference(
            label_names
        )
        labels = df_train[label_names].values.ravel()

        # Train the model on the training data
        self.model.fit(amplitudes, labels)

    def test(self, df_test: pd.DataFrame) -> float:
        """
        Test the model
        
        :param test_df: test dataframe
        
        :return: score of the model
        """
        # Get the the columns that start with 'Amplitude' and suffix with a
        # number
        amplitude_names = df_test.filter(regex='Amplitude\d+').columns
        amplitudes = df_test[amplitude_names].values

        # Get the column that start with 'Label' and no suffix
        label_names = df_test.filter(regex='Label\d+').columns
        label_names = df_test.filter(regex='Label').columns.difference(
            label_names
        )
        labels = df_test[label_names].values.ravel()

        # Test the model
        score = self.model.score(amplitudes, labels)

        # Return the score of the model
        return score

    def predict(self, df_predictions: pd.DataFrame) -> list[float]:
        """
        Predict the labels of the test data
        
        :param test_df: test dataframe
        :param regex: regex to use to get the columns
        
        :return: predictions
        """
        # Get the the columns that start with 'Amplitude' and suffix with a
        # number
        amplitude_names = df_predictions.filter(regex='Amplitude\d+').columns
        amplitudes = df_predictions[amplitude_names].values

        # Predict the labels
        predictions = self.model.predict_proba(amplitudes)

        # Return the predictions
        return predictions

from sklearn.neighbors import NearestCentroid as NearestCentroidClassifier

from models.IArtificialIntelligence import IArtificialIntelligence

class NearestCentroid(IArtificialIntelligence):
    def __init__(self):
        self.model = self.createModel()

    def createModel(self):
        """
        Create a k nearest neighbor model
        
        :return: model
        """

        model = NearestCentroidClassifier()

        return model

    def train(self, df_train, regex):
        """
        Train the model
        
        :param train_df: training dataframe
        :param regex: regex to use to get the columns
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_train.filter(regex=regex).columns
        amplitudes = df_train[amplitude_names].values

        # Get the column that start with 'Label' and no suffix
        label_names = df_train.filter(regex='Label\d+').columns
        label_names = df_train.filter(regex='Label').columns.difference(label_names)
        labels = df_train[label_names].values.ravel()

        # Train the model
        self.model.fit(amplitudes, labels)

    def test(self, df_test, regex):
        """
        Test the model
        
        :param test_df: test dataframe
        :param regex: regex to use to get the columns
        
        :return: score
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_test.filter(regex=regex).columns
        amplitudes = df_test[amplitude_names].values

        # Get the column that start with 'Label' and no suffix
        label_names = df_test.filter(regex='Label\d+').columns
        label_names = df_test.filter(regex='Label').columns.difference(label_names)
        labels = df_test[label_names].values.ravel()

        # Test the model
        score = self.model.score(amplitudes, labels)

        return score

    def predict(self, df_predictions, regex):
        """
        Predict the labels of the test data
        
        :param test_df: test dataframe
        :param regex: regex to use to get the columns
        
        :return: predictions
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_predictions.filter(regex=regex).columns
        amplitudes = df_predictions[amplitude_names].values

        # Predict the labels
        predictions = self.model.predict(amplitudes)

        return predictions

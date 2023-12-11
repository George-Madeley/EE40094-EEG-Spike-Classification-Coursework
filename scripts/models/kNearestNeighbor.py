from sklearn.neighbors import KNeighborsClassifier

from IArtificialIntelligence import IArtificialIntelligence

class KNearestNeighbor(IArtificialIntelligence):
    def __init__(self, k):
        self.model = self.createModel(k)

    def createModel(self, k):
        """
        Create a k nearest neighbor model
        
        :param k: number of neighbors
        
        :return: model
        """

        model = KNeighborsClassifier(n_neighbors=k)

        return model

    def train(self, df_train):
        """
        Train the model
        
        :param train_df: training dataframe
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_train.filter(regex='PC\d+').columns
        amplitudes = df_train[amplitude_names].values

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_train.filter(regex='Label').columns
        labels = df_train[label_names].values

        # Train the model
        self.model.fit(amplitudes, labels)

    def test(self, df_test):
        """
        Test the model
        
        :param test_df: test dataframe
        
        :return: score
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_test.filter(regex='PC\d+').columns
        amplitudes = df_test[amplitude_names].values

        # Get the columns that start with 'Label' and suffix with a number
        label_names = df_test.filter(regex='Label').columns
        labels = df_test[label_names].values

        # Test the model
        score = self.model.score(amplitudes, labels)

        return score

    def predict(self, df_predictions):
        """
        Predict the labels of the test data
        
        :param test_df: test dataframe
        
        :return: predictions
        """
        # Get the the columns that start with 'Amplitude' and suffix with a number
        amplitude_names = df_predictions.filter(regex='PC\d+').columns
        amplitudes = df_predictions[amplitude_names].values

        # Predict the labels
        predictions = self.model.predict_proba(amplitudes)

        return predictions

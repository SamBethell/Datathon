import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self):
        pass

    def correlation_finder(self, data):
        correlation = data.corr()
        specific_correlation = correlation.loc['pre_loans5']
        correlations = {}
        indices = []
        for value in range(0, len(specific_correlation)):
            values = specific_correlation.values[value]
            column = specific_correlation.index[value]
            indices.append(column)
            correlations[column] = values
        return correlations, indices

    def cleaning(self, indices, correlations, data):
        cleaned_columns = [col for col in indices if correlations[col] > 0.01]
        cleaned_data = data[cleaned_columns]
        return cleaned_data

    def training(self, df):  # df will be cleaned data as a DataFrame
        X_train = df.drop(columns=['pre_loans5'])
        Y_train = df['pre_loans5']
        model = LogisticRegression(max_iter=100, solver='liblinear')
        model.fit(X_train, Y_train)
        return model

    def predict(self, data, model):
        test_data = data.drop(columns=['pre_loans5'])
        return model.predict(test_data)




Model_instance = Model()

# Using all 10 data sets takes a very long time so only 1 dataset usef and 99.4% accuracy was achieved.

data = pd.read_csv("datathon_student/train_data/train_data_0.csv")
test_data = pd.read_csv("datathon_student/test_folder/test_data_0.csv")
correlations, indices = Model_instance.correlation_finder(data)
cleaned_data = Model_instance.cleaning(indices, correlations, data)
test_data = test_data[[col for col in cleaned_data]]
predictions = Model_instance.predict(test_data, Model_instance.training(cleaned_data))
accuracy = accuracy_score(test_data['pre_loans5'], predictions)
print(accuracy)






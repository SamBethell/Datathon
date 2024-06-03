import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# path is "datathon_student/train_data/train_data_0.csv"

class model:
    def __init__(self):
        poo = []
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
    def cleaning(self, indexes, correlations):
        cleaned_data = {}
        for value in range(0, len(indexes)):
            if correlations[indexes[value]] > 0.01:
                cleaned_data[indexes[value]] = correlations[indexes[value]]
        return cleaned_data
    def training(self, df): # df will be cleaned data
        X_train = df[[]]
        Y_train = df[['pre_loans5']]
        Y_train = np.ravel(Y_train)
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, Y_train)
        return model


Model = model()
for value in range(0, 11):
    data = pd.read_csv("datathon_student/train_data/train_data_0.csv")
    correlations, indices = Model.correlation_finder(data)
    cleaned_data = Model.cleaning(indices, correlations)
    Model = Model.training(cleaned_data)






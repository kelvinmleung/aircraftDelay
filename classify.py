import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Classify:

    def __init__(self):
        self.datadir = '../data/'

    def readTrainTest(self):
        X_train = pd.read_csv(self.datadir + 'X_train.csv')
        X_test = pd.read_csv(self.datadir + 'X_test.csv')
        y_train = pd.read_csv(self.datadir + 'y_train.csv')
        y_test = pd.read_csv(self.datadir + 'y_test.csv')

        return X_train, X_test, y_train, y_test
    
    def scaleData(self, X_train, X_test):
        print('Transforming data...')
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return X_train, X_test

    def fitPredictModel(self, clf, X_train, X_test, y_train, y_test):
        print('Fitting model...')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Accuracy:', accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print('Classification Report:\n', classification_report(y_test, y_pred))

    def runLogistic(self):
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = LogisticRegression(verbose=1)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)

    def runSVM(self):
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = SVC(C=50, verbose=1)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)

    def runNeuralNet(self):
        #(hidden_layer_sizes=100, activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = MLPClassifier(hidden_layer_sizes=(5,), verbose=1)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)
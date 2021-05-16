import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, _tree

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
        print('Transforming data...\n')
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return X_train, X_test

    def fitPredictModel(self, clf, X_train, X_test, y_train, y_test):
        print('Fitting model...\n')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        conf = confusion_matrix(y_test, y_pred)
        sens = conf[1,1] / (conf[1,1] + conf[1,0])
        spec = conf[0,0] / (conf[0,0] + conf[0,1])

        print('Confusion Matrix:\n', conf)
        print('\nSensitivity:', sens)
        print('Specificity:', spec)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('\nClassification Report:\n', classification_report(y_test, y_pred))

        return clf

    def runLogistic(self, C=1):
        print('\n### Logistic Regression ###')
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = LogisticRegression(C=C, verbose=1, random_state=1021)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)

    def runSVM(self, C=1):
        print('\n### SVM ###')
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = SVC(C=C, verbose=1, random_state=1021)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)

    def runNeuralNet(self):
        print('\n### Neural Network ###')
        #(hidden_layer_sizes=100, activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = MLPClassifier(hidden_layer_sizes=(5,), verbose=1, random_state=1021)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)

    def runTree(self):
        print('\n### Decision Tree ###')
        X_train, X_test, y_train, y_test = self.readTrainTest()
        clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=1021)
        clf = self.fitPredictModel(clf, X_train, X_test, y_train, y_test)
        # r = export_text(clf, feature_names=X_train.columns.values.tolist())
        # print(r)
        # tree_to_code(clf, feature_names=X_train.columns.values.tolist())

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
    # print("def predict({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

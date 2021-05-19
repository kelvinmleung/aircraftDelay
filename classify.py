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
from sklearn.model_selection import learning_curve

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scikitplot.metrics import plot_cumulative_gain

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
        # scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return X_train, X_test

    def fitPredictModel(self, clf, X_train, X_test, y_train, y_test, printResults=True):
        print('Fitting model...\n')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        conf = confusion_matrix(y_test, y_pred)
        acc, sens, spec = self.sensSpec(conf)

        if printResults == True:
            print('Confusion Matrix:\n', conf)
            print('\nSensitivity:', sens)
            print('Specificity:', spec)
            print('Accuracy:', acc)
            # print('Accuracy:', accuracy_score(y_test, y_pred))
            print('\nClassification Report:\n', classification_report(y_test, y_pred))

        return clf, conf
    
    def sensSpec(self, conf):
        acc = (conf[1,1] + conf[0,0]) / np.sum(conf)
        sens = conf[1,1] / (conf[1,1] + conf[1,0])
        spec = conf[0,0] / (conf[0,0] + conf[0,1])
        
        return acc, sens, spec

    def gainsChart(self, clf, X_train, y_train):
        predictions = clf.predict_proba(X_train)
        plot_cumulative_gain(y_train, X_train, figsize=(12, 8), title_fontsize=20, text_fontsize=18)

    def runLogistic(self, C=1):
        print('\n### Logistic Regression ###')
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = LogisticRegression(C=C, verbose=False, random_state=1021)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)
        self.gainsChart(clf, X_train, y_train)

    def tuneLogistic(self, C=[1e5,1e4,1e3,1e2,1e1,1,1e-1,1e-2]):
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        n = len(C)
        acc, sens, spec = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            clf = LogisticRegression(C=C[i], max_iter=10000, verbose=False, random_state=1021)
            clf, conf = self.fitPredictModel(clf, X_train, X_test, y_train, y_test, printResults=False)
            acc[i], sens[i], spec[i] = self.sensSpec(conf)
        plt.figure()
        plt.semilogx(C, acc, label='Accuracy')
        plt.semilogx(C, sens, label='Sensitivity')
        plt.semilogx(C, spec, label='Specificity')
        plt.xlabel('Regularization Parameter C')
        plt.title('Logistic Regression - Tuning C')
        plt.grid()
        plt.legend()

    

    def trainSubset(self, X_train, y_train, size):
        n = len(y_train)
        indKeep = np.random.randint(0, n, size=size)
        X = X_train.iloc[indKeep, :].reset_index(drop=True)
        y = y_train.iloc[indKeep, :].reset_index(drop=True)
        
        return X, y

    def runSVM(self, C=10):
        print('\n### SVM ###')
        X_train, X_test, y_train, y_test = self.readTrainTest()
        # varKeep = ['FLIGHTS_IN_DAY','DISTANCE','CRS_ELAPSED_TIME', 'CRS_DEP_TIME','CRS_ARR_TIME']
        # X_train = X_train[varKeep]
        # X_test = X_test[varKeep]

        X_train, y_train = self.trainSubset(X_train, y_train, size=10000)
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = SVC(C=C, verbose=False, random_state=1021)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)

    def tuneSVM(self, C=[1e2,1e1,1,1e-1,1e-2]):
        X_train, X_test, y_train, y_test = self.readTrainTest()
        # X_train, y_train = self.trainSubset(X_train, y_train, size=1000)
        # print(X_train)
        X_train, X_test = self.scaleData(X_train, X_test)
        n = len(C)
        
        acc, sens, spec = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            clf = clf = SVC(C=C[i], verbose=False, random_state=1021)
            clf, conf = self.fitPredictModel(clf, X_train, X_test, y_train, y_test, printResults=False)
            acc[i], sens[i], spec[i] = self.sensSpec(conf)
        plt.figure()
        plt.semilogx(C, acc, label='Accuracy')
        plt.semilogx(C, sens, label='Sensitivity')
        plt.semilogx(C, spec, label='Specificity')
        plt.xlabel('Regularization Parameter C')
        plt.title('SVM - Tuning C')
        plt.grid()
        plt.legend()

    def runNeuralNet(self, layers=(5,5), alpha=0.0001):
        print('\n### Neural Network ###')
        #(hidden_layer_sizes=100, activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        clf = MLPClassifier(hidden_layer_sizes=layers, alpha=alpha, activation='logistic', verbose=False, random_state=1021)
        self.fitPredictModel(clf, X_train, X_test, y_train, y_test)
        self.plot_learning_curve(clf, 'Learning Curve - Neural Net', X_train, y_train, axes=None, ylim=None, cv=None)

    def tuneNeuralNet(self, layers, alpha=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]):
        X_train, X_test, y_train, y_test = self.readTrainTest()
        X_train, X_test = self.scaleData(X_train, X_test)
        n = len(alpha)
        
        acc, sens, spec = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            clf = MLPClassifier(hidden_layer_sizes=layers, alpha=alpha[i], activation='logistic', verbose=False, random_state=1021)
            clf, conf = self.fitPredictModel(clf, X_train, X_test, y_train, y_test, printResults=False)
            acc[i], sens[i], spec[i] = self.sensSpec(conf)
        plt.figure()
        plt.semilogx(alpha, acc, label='Accuracy')
        plt.semilogx(alpha, sens, label='Sensitivity')
        plt.semilogx(alpha, spec, label='Specificity')
        plt.xlabel('Regularization Parameter Alpha')
        plt.title('Neural Network - Tuning Alpha, Layers=' + str(layers))
        plt.grid()
        plt.legend()

    def runTree(self, maxDepth=10, minLeaf=50, printRules=False):
        print('\n### Decision Tree ###')
        X_train, X_test, y_train, y_test = self.readTrainTest()
        clf = DecisionTreeClassifier(max_depth=maxDepth, min_samples_leaf=minLeaf, random_state=1021)
        clf, conf = self.fitPredictModel(clf, X_train, X_test, y_train, y_test)
        # r = export_text(clf, feature_names=X_train.columns.values.tolist())
        # print(r)
        # tree_to_code(clf, feature_names=X_train.columns.values.tolist())
        if printRules == True:
            rules = self.get_rules(clf, feature_names=X_train.columns.values.tolist(), class_names=['0', '1'])
            for r in rules:
                print(r)
        # if maxDepth <= 3:
            # plt.figure(figsize=(25,20))
        # plt.figure()
        plt.figure(figsize=(10,10))
        plot_tree(clf, max_depth=3, feature_names=X_train.columns.values.tolist(), class_names=['0','1'], filled=True)

    def tuneTree_acc(self, depth=list(range(1,11)), minLeaf=[2,5,10,20,50,100]):
        X_train, X_test, y_train, y_test = self.readTrainTest()
        for i in range(len(depth)):
            for j in range(len(minLeaf)):
                clf = DecisionTreeClassifier(max_depth=depth[i], min_samples_leaf=minLeaf[j], random_state=1021)
                clf, conf = self.fitPredictModel(clf, X_train, X_test, y_train, y_test, printResults=False)
                acc, sens, spec = self.sensSpec(conf)
                print('Depth:', depth[i])
                print('Min Sample in Leaf:', minLeaf[j])
                print('\tAccuracy:', acc)

    def tuneTree(self, depth=list(range(1,16)), minLeaf=50):
        X_train, X_test, y_train, y_test = self.readTrainTest()
        n = len(depth)
        acc, sens, spec = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            clf = DecisionTreeClassifier(max_depth=depth[i], min_samples_leaf=minLeaf, random_state=1021)
            clf, conf = self.fitPredictModel(clf, X_train, X_test, y_train, y_test, printResults=False)
            acc[i], sens[i], spec[i] = self.sensSpec(conf)
        plt.figure()
        plt.plot(depth, acc, label='Accuracy')
        plt.plot(depth, sens, label='Sensitivity')
        plt.plot(depth, spec, label='Specificity')
        plt.xlabel('Max Depth of Tree')
        plt.title('Tree - Tuning the Max Depth')
        plt.grid()
        plt.legend()


    def get_rules(self, tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []
        
        def recurse(node, path, paths):
            
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]
                
        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]
        
        rules = []
        for path in paths:
            rule = "if "
            
            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: "+str(np.round(path[-1][0][0][0],3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]
            
        return rules

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

    def plot_learning_curve(self, estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes,
                        return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                            fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt

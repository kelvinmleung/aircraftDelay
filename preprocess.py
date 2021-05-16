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


pd.set_option('display.max_columns', 30)

class Preprocess:

    def __init__(self, airport='BOS'):
        self.datadir = '../data/'
        self.airport = airport

    def parseData(self, filename):
        print('Loading data...')
        df = pd.read_csv(self.datadir + filename)

        # drop the variables that will not be considered
        df = df[['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST', 'DISTANCE', 'CRS_DEP_TIME', 
                    'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'ARR_DELAY', 'CANCELLED', 'DIVERTED']]

        print('Factoring delay variable...')
        # factor the delay variable and apply LOGICAL_OR with cancel/divert
        delay_factor = pd.cut(df.ARR_DELAY, [-9999, 15, 9999], right=False, labels=[0,1]).to_numpy()
        cancelDivert = np.logical_or(df.CANCELLED.to_numpy(), df.DIVERTED.to_numpy())
        df['ARR_DELAY'] = np.logical_or(delay_factor, cancelDivert).astype(int)
        df = df.drop(['CANCELLED', 'DIVERTED'], axis=1)

        # print(df.info())
        print('Number of Null Entries:')
        print(df.isnull().sum())

        df = df.dropna(axis=0)
        print('Deleted rows with null entries.')

        # analyze flight date
        print('\nAnalyzing flight date...')
        dayofweek, dayofyear, month, flightsinday = self.analyzeFlightDate(df.FL_DATE)  
        df['DAY_OF_WEEK'] = dayofweek.tolist()
        df['FLIGHTS_IN_DAY'] = flightsinday.tolist()

        df.to_csv(self.datadir + 'parsedData.csv', index=False)

    def analyzeFlightDate(self, fl_date):
        dates_str = fl_date.tolist()
        dates_nonrepeat = list(set(dates_str)) # list of dates in the year
        n = len(dates_str) # number of samples in data
        ndays = len(dates_nonrepeat) # number of days in the year
        flightsbyday = np.zeros(ndays) # number of scheduled flights on each calendar day

        # categorize the date in several ways
        dayofweek = np.zeros(n, dtype='U3')
        dayofyear = np.zeros(n)
        month = np.zeros(n)
        flightsinday = np.zeros(n)
        for i in range(ndays):
            flightsbyday[i] = dates_str.count(dates_nonrepeat[i])
        for i in range(n):
            y = int(dates_str[i][:4])
            m = int(dates_str[i][5:7])
            d = int(dates_str[i][8:])
            dayofweek[i] = datetime.date(y,m,d).strftime('%a')
            dayofyear[i] = datetime.date(y,m,d).strftime('%-j')
            month[i] = m
            flightsinday[i] = flightsbyday[int(dayofyear[i]-1)]
        return dayofweek, dayofyear, month, flightsinday

    def filterByAirport(self):
        df = pd.read_csv(self.datadir + 'parsedData.csv')

        print('Filtering airport...')
        n = max(df.count(axis=0))
        rowsToKeep = []
        for i in range(n):
            # if (df.ORIGIN[i] == 'BOS' and df.DEST[i] == 'JFK') or (df.ORIGIN[i] == 'JFK' and df.DEST[i] == 'BOS'):
            if df.ORIGIN[i] == self.airport or df.DEST[i] == self.airport:
                rowsToKeep = rowsToKeep + [i]

        df.iloc[rowsToKeep, :].to_csv(self.datadir + 'parsedData_' + self.airport + '.csv', index=False)

    def evenOutTrainData(self, X_train, y_train):
        # Makes the training data have a similar number of delayed vs non-delayed entries
        X_train  = X_train.reset_index(drop=True)
        y_train  = y_train.reset_index(drop=True)

        ylist = y_train.ARR_DELAY.tolist()
        length = len(ylist)
        num1 = ylist.count(1)
        num0 = length - num1
        indlist = y_train.index.to_numpy()
        indToRemove = []
        


        # for i in range(num0 - num1):
        while num0 > num1:
            rand = np.random.randint(0, length)
            if y_train.ARR_DELAY[rand] == 0 and (rand not in indToRemove):
                indToRemove = indToRemove + [rand]
                num0 = num0 - 1
        X_train = X_train.drop(labels=indToRemove, axis=0)
        y_train = y_train.drop(labels=indToRemove, axis=0)

        print(y_train)

        return X_train, y_train

    def createMLdf(self):

        df = pd.read_csv(self.datadir + 'parsedData_' + self.airport + '.csv')

        # filter by airport
        # print('Filtering airport...')
        # df = self.filterByAirport(df, airport='EWR')

        print(df.info())

        # convert time to hour of the day
        df['CRS_DEP_TIME'] = df.CRS_DEP_TIME.to_numpy() // 100
        df['CRS_ARR_TIME'] = df.CRS_ARR_TIME.to_numpy() // 100

        varML = ['DAY_OF_WEEK','FLIGHTS_IN_DAY','OP_CARRIER','ORIGIN','DEST','DISTANCE','CRS_DEP_TIME','CRS_ARR_TIME','CRS_ELAPSED_TIME']
        varEncode = ['DAY_OF_WEEK', 'OP_CARRIER','ORIGIN','DEST','CRS_DEP_TIME','CRS_ARR_TIME']
        varOther = ['FLIGHTS_IN_DAY', 'DISTANCE', 'CRS_ELAPSED_TIME']

        X_train, X_test, y_train, y_test = train_test_split(df[varML], df[['ARR_DELAY']], test_size=0.3,random_state=10)

        X_train = pd.concat([X_train[varOther], pd.get_dummies(X_train[varEncode])], axis=1)
        X_test = pd.concat([X_test[varOther], pd.get_dummies(X_test[varEncode])], axis=1)
        
        print('Evening out training samples...')
        X_train, y_train = self.evenOutTrainData(X_train, y_train)

        print(X_train)
        print(y_train)

        X_train.to_csv(self.datadir + 'X_train.csv', index=False)
        X_test.to_csv(self.datadir + 'X_test.csv', index=False)
        y_train.to_csv(self.datadir + 'y_train.csv', index=False)
        y_test.to_csv(self.datadir + 'y_test.csv', index=False)

    '''
    def createplotdf(self):
        df = pd.read_csv(self.datadir + 'parsedData.csv')

        # analyze flight date
        dayofweek, dayofyear, month, flightsinday = self.analyzeFlightDate(df.FL_DATE)  
        df['FLIGHTS_IN_DAY'] = flightsinday.tolist()
        df['DAY_OF_WEEK'], dayWeekLabel = pd.factorize(dayofweek)
        df['DAY_OF_YEAR'], dayYearLabel = pd.factorize(dayofyear)
        df['MONTH'], monthLabel = pd.factorize(month)
        
        # convert time to hour of the day
        df['CRS_DEP_TIME'] = df.CRS_DEP_TIME.to_numpy() // 100
        df['CRS_ARR_TIME'] = df.CRS_ARR_TIME.to_numpy() // 100

        # factorize carrier and origin/destination
        df['OP_CARRIER'], carrierLabel = pd.factorize(df.OP_CARRIER)
        df['ORIGIN'], originLabel = pd.factorize(df.ORIGIN)
        df['DEST'], destLabel = pd.factorize(df.DEST)

        # convert labels to numpy, save
        carrierLabel = carrierLabel.to_numpy(dtype=str)
        originLabel = originLabel.to_numpy(dtype=str)
        destLabel = destLabel.to_numpy(dtype=str)
        np.savetxt(self.datadir + 'days.txt', dayLabel)
        np.savetxt(self.datadir + 'carriers.txt', carrierLabel, fmt='%s')
        np.savetxt(self.datadir + 'origins.txt', originLabel, fmt='%s')
        np.savetxt(self.datadir + 'dests.txt', destLabel, fmt='%s')

        df.to_csv(self.datadir + 'dataPlot.csv')
    '''

    def barPlot(self, x, y, n, xlabels=None, varName=''):
        total = np.zeros(n)
        delayed = np.zeros(n)

        for i in range(len(x)):
            total[x[i]] = total[x[i]] + 1
            if y[i] == 1:
                delayed[x[i]] = delayed[x[i]] + 1
        prop = delayed / total

        plt.figure()
        plt.bar(list(range(n)), prop, tick_label=xlabels)
        plt.title('Flight Delays by ' + str(varName))
        plt.xlabel(varName)
        plt.ylabel('Proportion of Delayed Flights')


    def initialPlots(self):
        df = pd.read_csv(self.datadir + 'dataPlot.csv')
        dayofweek = df.DAY_OF_WEEK.to_numpy()
        dayofyear = df.DAY_OF_YEAR.to_numpy()
        month = df.MONTH.to_numpy()
        dep = df.CRS_DEP_TIME.to_numpy()
        arr = df.CRS_ARR_TIME.to_numpy()
        delay = df.ARR_DELAY.to_numpy()
        carrier = df.OP_CARRIER.to_numpy()
        carrierLabel = np.loadtxt(self.datadir + 'carriers.txt', dtype=str)

        self.barPlot(dayofweek, delay, 7, varName='Day of Week')
        self.barPlot(month, delay, 12, varName='Month')
        self.barPlot(dep, delay, 24, varName='Departure Hour')
        self.barPlot(arr, delay, 24, varName='Arrival Hour')
        self.barPlot(carrier, delay, len(carrierLabel), xlabels=carrierLabel, varName='Carrier')
        






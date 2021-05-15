import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 30)

class AircraftDelay:

    def __init__(self):
        self.datadir = '../data/'

    def parseData(self, filename):
        df = pd.read_csv(self.datadir + filename)

        # drop the variables that will not be considered
        df = df[['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST', 'DISTANCE', 'CRS_DEP_TIME', 
                    'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'ARR_DELAY', 'CANCELLED', 'DIVERTED']]

        # factor the delay variable and apply OR with cancel/divert
        delay_factor = pd.cut(df.ARR_DELAY, [-9999, 15, 9999], right=False, labels=[0,1]).to_numpy()
        cancelDivert = np.logical_or(df.CANCELLED.to_numpy(), df.DIVERTED.to_numpy())
        df['ARR_DELAY'] = np.logical_or(delay_factor, cancelDivert).astype(int)
        df = df.drop(['CANCELLED', 'DIVERTED'], axis=1)

        print(df.info())
        df.to_csv(self.datadir + 'parsedData.csv')

    def analyzeFlightDate(self, fl_date):
        dates_str = fl_date.tolist()
        dates_nonrepeat = list(set(dates_str)) # list of dates in the year
        n = len(dates_str) # number of samples in data
        ndays = len(dates_nonrepeat) # number of days in the year
        flightsbyday = np.zeros(ndays) # number of scheduled flights on each calendar day

        # categorize the date in several ways
        dayofweek = np.zeros(n)
        dayofyear = np.zeros(n)
        month = np.zeros(n)
        flightsinday = np.zeros(n)
        for i in range(ndays):
            flightsbyday[i] = dates_str.count(dates_nonrepeat[i])
        for i in range(n):
            y = int(dates_str[i][:4])
            m = int(dates_str[i][5:7])
            d = int(dates_str[i][8:])
            dayofweek[i] = datetime.date(y,m,d).strftime('%w')
            dayofyear[i] = datetime.date(y,m,d).strftime('%-j')
            month[i] = m
            flightsinday[i] = flightsbyday[int(dayofyear[i]-1)]
        return dayofweek, dayofyear, month, flightsinday

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
        np.savetxt('days.txt', dayLabel)
        np.savetxt('carriers.txt', carrierLabel, fmt='%s')
        np.savetxt('origins.txt', originLabel, fmt='%s')
        np.savetxt('dests.txt', destLabel, fmt='%s')

        df.to_csv(self.datadir + 'dataPlot.csv')

    def createMLdf(self):
        df = pd.read_csv(self.datadir + 'dataPlot.csv')


        df.to_csv(self.datadir + 'dataML.csv')

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
        carrierLabel = np.loadtxt('carriers.txt', dtype=str)

        self.barPlot(dayofweek, delay, 7, varName='Day of Week')
        self.barPlot(month, delay, 12, varName='Month')
        self.barPlot(dep, delay, 24, varName='Departure Hour')
        self.barPlot(arr, delay, 24, varName='Arrival Hour')
        self.barPlot(carrier, delay, len(carrierLabel), xlabels=carrierLabel, varName='Carrier')
        
        # dealing with categorical variables
        # https://www.kaggle.com/getting-started/55836
        # https://www.pluralsight.com/guides/handling-categorical-data-in-machine-learning-models (one hot encoding)
        # https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
        # https://www.kaggle.com/fabiendaniel/predicting-flight-delays-tutorial

    def runAlg():
        df = pd.read_csv(self.datadir + 'parsedData.csv')

        # scale the data first 
        X = StandardScaler()

        # split into test and train
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3,random_state=10) 


    








import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from preprocess import Preprocess
from classify import Classify 

pd.set_option('display.max_columns', 30)

ad = Preprocess(airport='EWR')
# ad.parseData('2017.csv')
# ad.filterByAirport()
# ad.createMLdf()

cl = Classify()
# cl.runLogistic(C=0.01)
# cl.runSVM(C=50)
# cl.runNeuralNet()
cl.runTree()


plt.show()

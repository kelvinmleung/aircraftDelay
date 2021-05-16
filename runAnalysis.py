import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from preprocess import Preprocess
from classify import Classify 

pd.set_option('display.max_columns', 30)

ad = Preprocess()
cl = Classify()
# ad.parseData('2017.csv')
# # initialPlots()
# ad.createMLdf()

# cl.runSVM()
# cl.runLogistic()
cl.runNeuralNet()


plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from preprocess import Preprocess
from classify import Classify 

pd.set_option('display.max_columns', 30)

airportList = ['ATL','LAX','ORD','DFW','DEN','JFK','SFO','LAS','SEA','CLT']
# airportList = ['LAX','BOS','ATL']
# airportList = ['ATL','LAX','ORD','DFW','DEN','JFK','SFO','LAS','SEA','CLT']#,  'EWR','MCO','PHX','MIA','IAH','BOS']#,'MSP','DTW','FLL','LGA']
ad = Preprocess(airportList=airportList)
# ad.parseData('2017.csv')

# create df for plotting before filtering out the airports
# ad.createplotdf()
# ad.initialPlots()

# filter by airport for the ML data
# ad.filterByAirport()
ad.createMLdf()


cl = Classify()

# tune the tree - choose depth = 10, minLeaf = 50
# cl.tuneTree_acc(depth=list(range(1,11)), minLeaf=[2,5,10,20,50,100])
# cl.tuneTree(depth=list(range(10,101,5)))
# cl.runTree(maxDepth=10, minLeaf=50, printRules=False)

# tune Logistic - choose C = 1
# cl.tuneLogistic()
# cl.runLogistic(C=1)

# tune SVM - choose C = 10
# cl.tuneSVM()
# cl.runSVM(C=10)

# tune NN - choose layers=(2,), alpha=0.01
# cl.tuneNeuralNet(layers=(2,))
# cl.tuneNeuralNet(layers=(5,))
# cl.tuneNeuralNet(layers=(2,2))
# cl.tuneNeuralNet(layers=(5,5))
# cl.runNeuralNet(layers=(5,5), alpha=0.01)









plt.show()

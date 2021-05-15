import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from aircraftDelay import AircraftDelay 

pd.set_option('display.max_columns', 30)

ad = AircraftDelay()
ad.parseData('2017.csv')
# initialPlots()

plt.show()

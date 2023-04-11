# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('code/arcanut.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values



#regressor = LinearRegression()
#regressor.fit(X, Y)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

regressor = LinearRegression()

regressor.fit(X_train, Y_train)
#predecting test set results

Y_pred = regressor.predict(X_test)
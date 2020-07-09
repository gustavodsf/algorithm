import time
from math import log, sqrt, trunc
from random import *

import matplotlib.pyplot as plt
import numpy as np

from scaler import Scaler

class LogisticRegression(object):
    
    def __init__(self):
        self.w = np.array([round(uniform(0,1), 2) for i in range(2)])
         
    def binaryCrossEntropy(self, ValorCorreto, ValorPredito):
        return -(ValorCorreto * np.log(ValorPredito)) + (1.0-ValorCorreto) * np.log(1.0-ValorPredito)

    def fit(self, x, y, alpha=0.1, iter = 4000):
        x = np.array(x)
        y = np.array(y)
        m = len(y)
        self.errorList = []
        for i in range(iter):
            answer = self.sigmoid(self.predict(x))
            self.errorList.append(self.binaryCrossEntropy(y, answer))
            sum1 = np.sum((self.sigmoid(self.predict(x)) - y) * x)
            sum2 = np.sum(self.sigmoid(self.predict(x)) - y)
            self.w[0] = self.w[0] - ((alpha / m)) * sum1
            self.w[1] = self.w[1] - ((alpha/m)) * sum2
    
    def checkHypothesis(self, x, y):
        x = np.array(x)
        y = np.array(y)     
        predicted = self.sigmoid(self.predict(x))
        plt.scatter(x, y)
        plt.plot(x, predicted)
        plt.show()

    def viewErrorChart(self):
        x = [i for i in range(len(self.errorList))]
        plt.plot(x, self.errorList)
        plt.show()
        
    def predict(self, x):
        return self.w[0] + self.w[1]*x
    
    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))

x = [1,2,3,4,5,12,13,14,15,16]
y = [0,0,0,0,0,1,1,1,1,1]

reg1 = LogisticRegression()
reg1.fit(x, y)
reg1.checkHypothesis(x,y)
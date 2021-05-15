import time
from math import log, sqrt, trunc
from random import *

import matplotlib.pyplot as plt
import numpy as np

from scaler import Scaler


class LinearRegression(object):
    
    def __init__(self):
        self.trained = False
        self.w0 = 0.1
        self.w1 = 0.1
        
    def predict(self, x):
        x = np.array(x)
        return self.w0 + self.w1 * x
    
    def fit(self, x, y, alpha=0.01, iter = 5000):
        x = np.array(x)
        y = np.array(y)
        self.trained = True
        self.eixoX = [i for i in range(0, iter)]
        size = len(x)
        self.error = []
        
        for i in range(0, iter):
            answer = self.predict(x)
            self.error.append(self._evaluate(answer, y))
            error_w0 = np.sum(answer-y)
            error_w1 = np.sum((answer-y) * x)            
            self.w0 = self.w0 - alpha * (1/float(size)) * error_w0
            self.w1 = self.w1 - alpha * (1/float(size)) * error_w1

        
    def _evaluate(self, answer, y):
        mse = 0
        for i in range(0, len(answer)):
            mse += (answer[i] - y[i])**2
        mse = mse / len(answer)
        return mse
    
    def checkHypothesis(self, x, y, title="Chart"):
        x = np.array(x)
        hypothesis = self.w0 + self.w1*x
        plt.scatter(x, y)
        plt.plot(x, hypothesis)
        plt.title(title)
        plt.show()

x = [1, 3, 5, 10, 13, 15]
y = [0, 0, 0, 1, 1, 1]

reg1 = LinearRegression()
reg1.fit(x, y)
reg1.checkHypothesis(x,y)
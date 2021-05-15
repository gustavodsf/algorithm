from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

class KNN(object):

    def fit(self, x, y, k=3):
        self.x = x
        self.y = y
        self.k = k
    
    def classify(self, point):
        
        nearPosition = []
        nearClass = []
        amountNearClass = []
        
        dist = self.calculateEuclidianDistance(self.x, point)
        distOrdered = sorted(dist)
        
        for i in range(self.k):
            for j in range(len(dist)):
                if distOrdered[i] == dist[j]:
                    nearPosition.append(j)
                    dist[j] = -1
        
        for i in range(0, self.k):
            nearClass.append(self.y[nearPosition[i]])
            
        unique, counts = np.unique(nearClass, return_counts=True)
        countClass = dict(zip(unique, counts))
        
        max =  -10
        pointClass = -1
        for key, number in countClass.items():
            if max < number:
                max = number
                pointClass = key
        return pointClass

    def calculateEuclidianDistance(self,x, point):
        dist = np.array(x) - np.array(point)
        dist = dist**2
        dist = np.array(list(map(lambda x: sqrt(sum(x)), dist)))
        
        return dist


x = [[1, 2, 4],
     [2, 1, 4],
     [1, 1, 4],
     [2, 2, 4],
     [4, 4, 4],
     [5, 4, 4],
     [3, 5, 4],
     [5, 6, 4]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

knn = KNN()
knn.fit(x, y)
result = knn.classify([4, 4, 4])

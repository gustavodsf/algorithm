from math import trunc

class Scaler(object):
    
    @staticmethod
    def absoluteScaler(input):
        map_list = list(map(lambda x: len(str(trunc(x))), input))
        biggest = sorted(map_list, reverse=True)[0]       
        absolute_value = 10**biggest
        return list(map(lambda x: x/absolute_value, input))
    
    @staticmethod
    def minMaxScaler(input):
        minValue = min(input)
        maxValue = max(input)
        return list(map(lambda x: (x - minValue)/(maxValue-minValue), input))
        

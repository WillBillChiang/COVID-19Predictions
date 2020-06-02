import requests
import pandas as pd
from consts import Consts
import numpy as np
from logistic import LogisticModel
from lstm import LSTMModel
from data import DataAnalyzer

'''
Main Code to Train and Run Models
'''

da = DataAnalyzer()
da.reportData()
totalUSCases = np.array([da.getTotal(da.data, t=i) for i in range(1, len(da.data.columns))])

modelType = input("logistic or lstm ")

if modelType == "logistic":
    model = LogisticModel(totalUSCases)
    model.trainLogistic()
    model.graph()
elif modelType == "lstm":
    daysIntoFuture = int(input("Input Number of Days Into the Future "))
    sampleSize = int(input("Input Sample Size "))
    epochs = int(input("Input Number of Epochs "))
    units = int(input("Input Number of LSTM Units "))
    LSTM = LSTMModel(totalUSCases, daysIntoFuture, sampleSize, units, epochs)
    LSTM.model() 
else:
    print("Invalid")
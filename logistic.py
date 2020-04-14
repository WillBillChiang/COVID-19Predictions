import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class LogisticModel():

    def __init__(self, cases):
        '''
        Initializes Object

        Args: 1-D array of cases at each time step
        '''
        self.parameters = np.random.logistic(size = 3)
        absArr = np.vectorize(abs)
        self.parameters = absArr(self.parameters/(max(self.parameters)))
        print(self.parameters)
        self.x = np.array([i for i in range(len(cases))])
        self.y = np.array(cases)

    def logistic(self, t, a, b, c):
        '''
        Logistic function for training

        Args: The time of the logistic prediction and the parameters
        Returns: Output of logistic function
        '''
        return c / (1 + a * np.exp(-b*t))

    def trainLogistic(self):
        '''
        Trains logistic growth model
        '''
        bounds = (0, [1000000000, 5, 1000000000])
        self.parameters, covariance = curve_fit(self.logistic, self.x, self.y, bounds=bounds, p0=self.parameters)

    def predict(self, t):
        '''
        Logistic function for graphing and predictions

        Args: The time of the logistic prediction
        Returns: Output of logistic function
        '''
        return self.parameters[2] / (1 + self.parameters[0] * np.exp(-self.parameters[1]*t))

    def graph(self):
        '''
        Graphs the data with logistic model
        '''
        plt.scatter(self.x, self.y)
        predictArr = np.vectorize(self.predict)
        plt.plot(self.x, predictArr(self.x))
        plt.title('Logistic Model Predictions')
        plt.show()
        plt.scatter(self.x, self.y)
        predictArr = np.vectorize(self.predict)
        graphX = np.append(self.x, [i for i in range(len(self.x), len(self.x)+100)])
        plt.plot(graphX, predictArr(graphX))
        plt.title('Logistic Model Predictions')
        plt.show()

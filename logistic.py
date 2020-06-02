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
        self.parameters = np.random.exponential(size = 3)
        print("Parameter Initialization")
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
        bounds = (0, [1e10, 10, 1e10])
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
        graphX = np.append(self.x, [i for i in range(len(self.x), len(self.x)*2)])
        numOfDays = 0
        for x in range(len(graphX[len(self.x):])):
            if self.predict(graphX[len(self.x) + x]) > self.parameters[2]*(0.999):
                numOfDays = x + 1
                break
        if (numOfDays == 0):
            numOfDays = "More Than " + str(len(self.x))
        plt.plot(graphX, predictArr(graphX))
        plt.title('Logistic Model Predictions | Max at ' + str("%.1f" % self.parameters[2]) + " Cases\nReached in " + str(numOfDays) + " Days") 
        plt.ylabel("Number of Cases")
        plt.xlabel("Days")
        print("Final Parameters")
        print(self.parameters)
        plt.show()

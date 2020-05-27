import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

DAYS_INTO_FUTURE = 10
SAMPLE_SIZE = 30

def split(cases):
        dataX = []
        dataY = []
        for i in range(len(cases)-(SAMPLE_SIZE+DAYS_INTO_FUTURE)):
            dataX.append(cases[i:i+SAMPLE_SIZE])
            dataY.append(cases[i+(SAMPLE_SIZE+DAYS_INTO_FUTURE)])
        return np.array(dataX), np.array(dataY)

class LSTMModel():
    def __init__(self, cases):
        self.__dataX, self.__dataY = split(cases)
        self.__modelInput = cases[-SAMPLE_SIZE:]
        print(self.__dataX, self.__dataY)
    
    def model(self):
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(SAMPLE_SIZE, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        X = self.__dataX.reshape((len(self.__dataX), SAMPLE_SIZE, 1))

        model.fit(X, self.__dataY, epochs=5000, verbose=1)

        modelInput = self.__modelInput.reshape(1, SAMPLE_SIZE, 1)
        prediction = model.predict(modelInput, verbose=0)
        print(prediction)
        

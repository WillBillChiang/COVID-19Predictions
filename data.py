import requests
import pandas as pd
from consts import Consts
import numpy as np
from logistic import LogisticModel
import lstm


class DataAnalyzer:


    def __init__(self):
        '''
        Fetches data from github and sets all base variables.

        Args: None
        Returns: None
        '''

        df = pd.read_csv(Consts.DATA_URL.value) # Read the URL
        df.drop(Consts.DROP_COLS.value, inplace=True, axis=1) # Drop columns

        self.data = self.cleanData(df)
        self.totalCases = self.getTotal(df)
        self.stateOrder = self.data.index




# ―――――――――――――――――――――――――――――――――――――――――――DATA PROCESSING――――――――――――――――――――――――――――――――――――――――――――――――――――――――

    def cleanData(self, df):
        '''
        Aggregates data for the number of cases for each state.

        Args: DataFrame with non-aggregated covid19 data
        Returns: DataFrame with the aggregated data
        '''

        aggregation_functions = {}
        for col in df.columns[1:]:
            aggregation_functions[col] = 'sum'

        df_new = df.groupby(df['Province_State']).aggregate(aggregation_functions)

        return df_new

    def getTotal(self, df, t=-1):
        '''
        Gets the total number of cases in the US.

        Args: DataFrame with the number of cases in each state and time step
        Returns: Number of cases in the US
        '''

        return sum(df.iloc[:, t])

# ―――――――――――――――――――――――――――――――――――――――――――DISPLAY DATA――――――――――――――――――――――――――――――――――――――――――――――――――――――――
    def hLine(self):
        '''
        Prints a horizontal Bar

        Args: None
        Returns: Horizontal Bar
        '''

        return '―' * 120

    def dLine(self):
        '''
        Prints a horizontal dashed line

        Args: None
        Returns: horizontal dashed line
        '''

        return '-' * 120

    def reportData(self):
        '''
        Prints all the data and variables in a clean way

        Args: None
        Returns: None
        '''

        vars = {
            'Data' : self.data.head(),
            'Total Cases' : self.totalCases,
            'State Order' : self.stateOrder,
        }

        for key in vars.keys():
            print(key + ":")
            print(self.dLine())
            print(vars[key])
            print(self.hLine())




if __name__ == '__main__':
    da = DataAnalyzer()
    da.reportData()

    totalUSCases = np.array([da.getTotal(da.data, t=i) for i in range(1, len(da.data.columns))])
    print(totalUSCases)
    # model = LogisticModel(totalUSCases)
    # model.trainLogistic()
    # model.graph()

    LSTM = lstm.LSTMModel(totalUSCases)
    LSTM.model()

# df['State']

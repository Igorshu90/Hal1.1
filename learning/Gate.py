# --------------------------------- Gate ---------------------------
# Parameters:   trading set
# Return:       pandas DataFrame after precessing
# -----------------------------------------------------------------------------

import pandas as pd
from learning.iopreprocessor import IoPreprocessor
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


class Gate:
    def __init__(self):
        self.classifiers = []
        self.feature_names = ['hour', 'lastCigTime', 'lastCigReason']
        self.target = 'Cause'
    def acquired_data(self):

        return pd.read_csv('/home/igor/Desktop/bla3.csv')

    def split_df_by_cause(self, df):
        df_list = []
        causes = np.unique(df[self.target].values, axis=0)
        for i in causes:
            df_list.append(df.loc[df[self.target] == i])
        return df_list

    def setclassifiers(self, df):
        iop = IoPreprocessor(df)
        df = iop.data_frame_normalizer()
        dflist = self.split_df_by_cause(df)

        for i in range(len(dflist)):
            classifier = IsolationForest(n_estimators=200, max_samples=50)
            x = dflist[i][self.feature_names]
            classifier.fit(x)
            self.classifiers.append(classifier)

    def predict(self, cig):
        df = self.acquired_data()
        df.loc[len(df)] = cig
        iop = IoPreprocessor(df)
        df = iop.data_frame_normalizer()
        x = df[self.feature_names].tail(1)
        i = df[self.target].tail(1).values[0]  # get index of write classifier
        return self.classifiers[i].predict(x)

    # this mathod is only for testing
    def predict_testv(self ):
        pass


def main():
    #
    # df = pd.read_csv('/home/igor/Desktop/bla3.csv')
    # row = ['137','2019-05-13','0 days 15:39:36.000000000','ww']
    # df.loc[len(df)] = row
    # print(df.tail())

    g = Gate()
    df = g.acquired_data()
    g.setclassifiers(df)
    print(g.predict(['139', '2019-05-13', '0 days 15:39:36.000000000', 'a']))

    iop = IoPreprocessor(df)
    df2 = iop.data_frame_normalizer()
    # print(df2.to_string())
    n = 4

    dfn = df2.loc[df2['Cause'] == n]
    dfn['a-scour'] = g.classifiers[n].predict(dfn[['hour', 'lastCigTime', 'lastCigReason']])

    df_normal = dfn.loc[dfn['a-scour'] == 1]
    df_outliyer = dfn.loc[dfn['a-scour'] == -1]
    plt.scatter(df_normal[ 'hour'], df_normal['lastCigReason'])
    plt.scatter(df_outliyer['hour'], df_outliyer['lastCigReason'], color='red')
    plt.show()








main()

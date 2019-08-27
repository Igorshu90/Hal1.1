# --------------------------------- Io_preprocessor ---------------------------
# Parameters:   pandas DataFrame (row Data)
# Return:       pandas DataFrame after precessing
# -----------------------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
import copy

class IoPreprocessor:

    def __init__(self, df):
        self.df = copy.deepcopy(df)


    def add_day_in_week(self):
        df1 = self.df['DateResample'].values
        res = np.zeros(len(df1))
        for i in range(len(df1)):
            date = df1[i].split("-")
            res[i] = datetime.datetime(int(date[0]), int(date[1]), int(date[2])).weekday()
        self.df['week_day'] = res

    def add_time_deltas(self):

        lastCigTime = 'lastCigTime'
        lastCigReason = 'lastCigReason'

        dates = self.df['DateResample'].values
        times = self.df['TimeResample'].values
        n = len(times)
        for i in range(n):
            times[i] = times[i][7:12]
        datetimes = []

        for i in range(n):
            datetimes.append(datetime.datetime.strptime(dates[i] + " " + str(times[i]), '%Y-%m-%d %H:%M'))

        causes = self.df['Cause']
        res = [9999]
        for curr in range(1, len(causes)):
            i = curr - 1

            while i != 0 and causes[curr] != causes[i]:
                i -= 1
            if i == 0:
                res.append(9999)
            else:
                res.append((datetimes[curr] - datetimes[i]).total_seconds() / 3600)
            curr += 1

        self.df[lastCigTime] = res
        res2 = [24]
        for i in range(1, n):
            res2.append((datetimes[i] - datetimes[i - 1]).total_seconds() / 3600)

        self.df[lastCigReason] = res2
        self.df = self.df.drop(self.df[self.df[lastCigTime] == 9999].index)

    def drop_redundant_columns(self):
        #self.df = self.df[['DateResample','hour', 'week_day', 'Cause']]
        pass


    def lable_code(self):
        pd.options.mode.chained_assignment = None
        le = LabelEncoder()
        self.df['Cause'] = le.fit_transform(self.df['Cause'])
        t = self.df['Cause'].values
        self.df['Cause'] = t

    def normalizer_time(self):
        df2 = self.df['TimeResample'].values
        res = np.zeros(len(df2))

        for i in range(len(df2)):
            res1 = float(df2[i][6:9])
            res2 = float(df2[i][10:12]) / 60
            res[i] = res1 + res2
        self.df['hour'] = res


    def data_frame_normalizer(self):
        self.add_day_in_week()
        self.normalizer_time()
        self.lable_code()
        self.drop_redundant_columns()
        self.add_time_deltas()
        return copy.deepcopy(self.df)




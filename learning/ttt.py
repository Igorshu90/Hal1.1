import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import  numpy as np
import matplotlib.pyplot as plt
import sys

# classifier
df = pd.read_csv('/home/igor/Desktop/subject_02.csv')

le_time = LabelEncoder()
le_rison = LabelEncoder()
t = LabelEncoder()
df['time_n'] = le_rison.fit_transform(df['time'])

df['rison_n'] = le_rison.fit_transform(df['rison'])
inputs = df.drop(['rison','time'],axis='columns')



anomalies_ratio = 0.5
if_sk = IsolationForest(n_estimators = 200,
                        max_samples = 10,
                        contamination = anomalies_ratio,
    )
print(inputs)
# plt.figure(figsize = (20,10))
# plt.scatter(inputs['time_n'],inputs['rison_n'])
# plt.show()

if_sk.fit(inputs)
y = if_sk.predict([[27, 2211]])
print(y)




#from prototype
#def cigToday(df):
#     cont = 0
#     for i in df :
#         if i == datetime.today():
#             cont +=1
#     return cont
#
# def addCig():
#
#     df = pd.DataFrame(con(msg='1'))
#     df = pd.to_datetime(df['DateResample'])
#     print(cigToday(df))
#     sigSunDay = 0
#     for i in df:
#         if i.weekday_name == 'Sunday':
#             sigSunDay +=1
#
#     df1 = df.drop_duplicates()
#     ave = sigSunDay/len(df1)
#
#     if (cigToday(df)>ave):
#         print('not allow')
#     else:
#         print('allow')
#
#
# #addCig()

def max3(x, y, z ):
    if x >= y and x>=z:
        return x
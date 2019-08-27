import pandas as pd
from learning.iopreprocessor import IoPreprocessor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib as mpl
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from scipy import stats
import time
from sklearn.decomposition import PCA
from sklearn import preprocessing

df = pd.read_csv('/home/igor/Desktop/bla3.csv')
df = IoPreprocessor(df).data_frame_normalizer

df1 = df.loc[df['Cause'] == 2]
df1 =df1[[  'id','week_day','hour'  ,  'test_at' , 'test_at2','Cause']]
df2 = df.loc[df['Cause'] == 0 ]
z1 = np.abs(stats.zscore(df1[[  'hour'  ,  'test_at' , 'test_at2']]))
z2 = np.abs(stats.zscore(df2[[    'hour'  ,  'test_at' , 'test_at2']]))

scaled_data = preprocessing.scale(df1[[   'week_day','hour'  ,  'test_at' , 'test_at2']])



plt.scatter(df['test_at2'],df['test_at'])
plt.show()
anomalies_ratio = 0.05

















if_sk = IsolationForest(n_estimators = 800,
                            max_samples = 50,

        )
if_sk.fit(df1[['hour','test_at','test_at2']])
df1['a-scour'] = if_sk.predict(df1[['hour','test_at','test_at2']])


df1_normal = df1.loc[df1['a-scour'] == 1]
df1_outliyer = df1.loc[df1['a-scour'] == -1]

# print(df1_outliyer)


clf = IsolationForest(n_estimators = 800,
                        max_samples = 50,)

clf.fit(df1_normal)
print(clf.predict(df1_outliyer))



plt.scatter(df1_normal['hour'],df1_normal['test_at'])
plt.scatter(df1_outliyer['hour'],df1_outliyer['test_at'] ,color='red')
plt.show()

print(df['Cause'])

# #########################################################################################################
df2=df

clf.fit(df2[['hour','test_at','test_at2','Cause']])
res1 = clf.predict(df2[['hour','test_at','test_at2','Cause']])

print()

X = df2[['hour','test_at','test_at2','Cause']].values
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[3])

X = ohe.fit_transform(X).toarray()


clf.fit(X)
res2 = clf.predict(X)
df2['a-scour'] = clf.predict(df2[['hour','test_at','Cause']])
df2 = df2.loc[df2['Cause'] == 0]


df2_normal = df2.loc[df2['a-scour'] == 1]
df2_outliyer = df2.loc[df2['a-scour'] == -1]


print(df2_outliyer.to_string())

plt.scatter(df2_normal['hour'],df2_normal['Cause'])
plt.scatter(df2_outliyer['hour'],df2_outliyer['Cause'] ,color='red')
plt.show()




# X = df2[['hour','test_at','Cause']]
# y=df2['a-scour']


#
# # Model (can also use single decision tree)
# from sklearn.ensemble import RandomForestClassifier
# model = IsolationForest(n_estimators=10)
#
# # Train
# model.fit(X)
#
# # Extract single tree
# estimator = model.estimators_[5]
#
# from sklearn.tree import export_graphviz
# # Export as dot

# export_graphviz(estimator, out_file='tree.dot',
#                 feature_names = ['hour','test_at','Cause'],
#                 class_names = ['-1','1'],
#                 rounded = True, proportion = False,
#                 precision = 2, filled = True)
#
# # Convert to png using system command (requires Graphviz)
# from subprocess import call
# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#
# # Display in jupyter notebook
# from IPython.display import Image
# Image(filename = 'tree.png')



df['a-scour'] = res2
df_normal = df.loc[df['a-scour'] == 1]
df_outliyer = df.loc[df['a-scour'] == -1]
print(df_outliyer.to_string())
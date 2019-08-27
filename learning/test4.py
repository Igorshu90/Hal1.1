import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from learning.iopreprocessor import IoPreprocessor

import os

df = pd.read_csv('/home/igor/Desktop/bla3.csv')
df = IoPreprocessor(df).data_frame_normalizer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D



from sklearn.ensemble import IsolationForest
clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12), \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(df[[   'hour'  ,  'test_at' , 'test_at2']])

pred = clf.predict(df[[   'hour'  ,  'test_at' , 'test_at2']])
df['anomaly']=pred
outliers=df.loc[df['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(df['anomaly'].value_counts())






from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(df[[   'hour'  ,  'test_at' , 'test_at2']])

res=pd.DataFrame(pca.transform(df[[   'hour'  ,  'test_at' , 'test_at2']]))

Z = np.array(res)

plt.title("IsolationForest")
plt.contourf( Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(res[0], res[1], c='green',
                 s=20,label="normal points")

b1 =plt.scatter(res.iloc[outlier_index,0],res.iloc[outlier_index,1], c='green',s=20,  edgecolor="red",label="predicted outliers")
plt.legend(loc="upper right")
plt.show()
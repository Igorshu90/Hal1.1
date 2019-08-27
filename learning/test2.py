import pandas as pd
from learning.iopreprocessor import IoPreprocessor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib as mpl
import pandas as pd
import matplotlib as mpl
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

df = pd.read_csv('/home/igor/Desktop/bla3.csv')
print(df)
df = IoPreprocessor(df).data_frame_normalizer

# total=0
# for i in range(100):

   # print(df.to_string())
clf = RandomForestClassifier(n_estimators=100,max_depth=3)
#    train, test = train_test_split(df, test_size=0.2)



   # # df = df.iloc[5:]
   #
   #  clf.fit(train[['week_day','hour','test_at','test_at2']],train['Cause'])
   #  y_pred = clf.predict(test[['week_day','hour','test_at','test_at2']])
   #  test_val = test['Cause'].values
   #  cont =0
   #  test['real'] = test_val
   #  test['pre'] = y_pred
   #
   #  for i in range(len(test_val)):
   #
   #      if test_val[i] == y_pred[i]:
   #          cont +=1
   #  print(test.to_string())
   #  print(cont/len(test_val))
   #
   #
   #  cont = 0
   #  print('---------------------')



X = df[['week_day','hour','test_at','test_at2']]
y = df['Cause']
scores = cross_val_score(clf, X  , y )
print(scores)

#############################################################################################################

from sklearn.datasets import load_iris
iris = load_iris()

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(X, y)

# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names = ['week_day','hour','test_at','test_at2'],
                class_names = ['0','1','2','3','4'],
                rounded = True, proportion = False,
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


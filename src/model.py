import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

df=pd.read_csv('../data/data.csv')

y=df.Species.copy()
X=df.copy()
X.drop(['Species'], axis=1, inplace=True)

clf = GaussianNB()
clf.fit(X,y)

joblib.dump(clf,'../model/clf.pkl')
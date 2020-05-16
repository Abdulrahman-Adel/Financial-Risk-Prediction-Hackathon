# -*- coding: utf-8 -*-
"""
Created on Sat May 16 07:14:04 2020

@author: Abdelrahman
"""

import pandas as pd

df_train = pd.read_csv("Train.csv")
df_test = pd.read_csv("Test.csv")


X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

X.drop(columns=["Past_Results","Loss_score","City"],inplace=True)
df_test.drop(columns=["Past_Results","Loss_score","City"],inplace=True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)

print(rfc.score(X_test, y_test))
print(log_loss(y_test, rfc.predict(X_test)))
print(f1_score(y_test, rfc.predict(X_test)))

    
importance = rfc.feature_importances_

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


import matplotlib.pyplot as plt

plt.bar([x for x in range(len(importance))], importance)
plt.show()




from xgboost import XGBClassifier


xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print(xgb.score(X_test, y_test))
print(log_loss(y_test, xgb.predict(X_test)))
print(f1_score(y_test, xgb.predict(X_test)))

    
importance2 = xgb.feature_importances_

for i,v in enumerate(importance2):
	print('Feature: %0d, Score: %.5f' % (i,v))


import matplotlib.pyplot as plt

plt.bar([x for x in range(len(importance2))], importance2)
plt.show()


XGB = XGBClassifier()
XGB.fit(X,y)
y_pred = XGB.predict_proba(df_test)

Y = pd.DataFrame(y_pred)

Y.to_excel("output1.xlsx",index=False)




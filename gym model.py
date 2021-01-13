# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:13:30 2021

@author: hp
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df=pd.read_excel(r'C:\Users\hp\Downloads\dataGYM.xlsx')


gym=df.copy()
del gym['BMI']
del gym['Class']
label_encoder = LabelEncoder()
gym['Prediction'] = label_encoder.fit_transform(df['Prediction'])


X = gym.iloc[:,:-1]
y = gym.iloc[:,-1]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

model_gym = RandomForestClassifier(n_estimators=20)

model_gym.fit(X_train, y_train)

print(model_gym)

expected = y_test
predicted = model_gym.predict(X_test)
# summarize the fit of the model
#Correction
metrics.classification_report(expected, predicted)
metrics.confusion_matrix(expected, predicted)



print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))



import pickle

pickle.dump(model_gym, open("model_gym.pkl", "wb"))

model = pickle.load(open("model_gym.pkl", "rb"))


print(model.predict([[40,5.6,70]]))



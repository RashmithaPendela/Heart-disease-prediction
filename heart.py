import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
df=pd.read_csv("heart.csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
columns_to_scale=["age","trestbps","chol","thalach","oldpeak"]
df[columns_to_scale]=sc.fit_transform(df[columns_to_scale])

x=df.iloc[:,:-1]
y=df['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

model1=LogisticRegression()
model1.fit(x_train,y_train)
y_pred=model1.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
conf_mat = confusion_matrix(y_test,y_pred)

import pickle
with open( 'log_model.pkl', 'wb') as f:
    pickle.dump(model1,f)

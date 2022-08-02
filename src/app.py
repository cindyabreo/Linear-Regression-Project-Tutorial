!pip install sklearn

#Importing necessary libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Step 1:

data = pd.read_csv('medical_insurance_cost.csv')

#step 2
data.info()

data.describe()

data.describe(include='O')

data.head(10)
#bmi indice de masa muscular

sb.heatmap(data.corr(), annot=True)

data.duplicated().any()

data[data.duplicated(keep=False)]

df=data.drop_duplicates()
df.info()

#creamos una nueva variable para las dummy
df=pd.get_dummies(df, columns=['sex','smoker','region'], drop_first=True)

#Step 3:

x=df.drop('charges', axis=1)
y=df['charges']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.25, random_state=15) 

modelo = LinearRegression()
modelo.fit(X_train, Y_train)
print("itercept:", modelo.intercept_)
print("coef:", modelo.coef_)
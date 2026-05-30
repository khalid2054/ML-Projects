import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('insurance_cleaned.csv')
# print(df.head(10))

X = df.drop('charges',axis=1)
Y = df['charges']

X_train,X_test, Y_train,Y_test = train_test_split( X, Y, test_size=0.20, random_state=42)
model = LinearRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print( 'mae' , mae)
print('mse', mse)
print('r2',r2)
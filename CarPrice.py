import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error,r2_score

df =pd.read_csv('ford.csv')

# data cleaning 
print(df.head(10))
print(df.isna().sum())
df = df[df['year']!= 2060]
print(df['year'].value_counts())
print(df['transmission'].value_counts())
print(df['fuelType'].value_counts())


# plot for information and understanding
sns.barplot(x='transmission',y = 'price',data=df)
plt.show()
sns.barplot(x='fuelType',y = 'price',data=df)
plt.show()
sns.barplot(x='engineSize',y = 'price',data=df)
plt.show()
# check correlation between varible or weight
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()
df_cleaned = df.copy()

# # encode
df_cleaned =pd.get_dummies(data=df,columns=['model','fuelType','transmission'],drop_first=True,dtype=int)
# print(df_cleaned.head(10))



# # for model......1
# # data splitting
X = df_cleaned.drop('price',axis =1)
y = df_cleaned['price']
X_train,X_test ,y_train,y_test = train_test_split( X, y, test_size=0.30, random_state=42)

# # satndardize data
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# # model 
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)


# # test result 
mae = mean_absolute_error(y_test, y_pred)
print(f'mae is model 1 {mae}')
mse = mean_squared_error(y_test, y_pred)
print(f'mse is model 1{mse}')
r2 = r2_score(y_test, y_pred)
print(f'r2 is model 1:  {r2}') 




# for model.....2
X2 = df_cleaned[['year','mileage','fuelType_Electric',  'model_ EcoSport', 
                 'model_ Edge', 'model_ Escort', 'model_ Fiesta',
       'model_ Focus', 'model_ Fusion', 'model_ Galaxy', 'model_ Grand C-MAX',
       'model_ Grand Tourneo Connect', 'model_ KA', 'model_ Ka+',
       'model_ Kuga', 'model_ Mondeo', 'model_ Mustang', 'model_ Puma',
       'model_ Ranger', 'model_ S-MAX', 'model_ Streetka',
       'model_ Tourneo Connect', 'model_ Tourneo Custom',
       'model_ Transit Tourneo', 'model_Focus','transmission_Semi-Auto',
       'transmission_Manual','engineSize','fuelType_Hybrid','fuelType_Other','fuelType_Petrol']]
y2 = df_cleaned['price']

X2_train, X2_test, y2_train, y2_test = train_test_split( X2, y2, test_size=0.30, random_state=42)


# standardize value for second model
scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)

# model 
model_2 = LinearRegression()
model_2.fit(X2_train,y2_train)
y_pred2 = model_2.predict(X2_test)


mae2 = mean_absolute_error(y2_test,y_pred2)
print(f'mae  model 2: {mae2}' )
mse2 = mean_squared_error(y2_test,y_pred2)
print(f'mse model 2 : {mse2}')
r_r2 =r2_score(y2_test,y_pred2)
print(f'r2 core model 2: {r_r2}')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score 

data = {
    'age': [19, 25, 31, None, 45, 52, 23, 40],
    'sex': ['female', 'male', 'female', 'male', 'female', 'male', None, 'female'],
    'bmi': [27.9, 30.1, None, 22.4, 35.3, 28.5, 26.2, 31.7],
    'children': [0, 1, 2, 1, 3, 2, None, 1],
    'smoker': ['yes', 'no', 'no', 'no', 'yes', 'yes', 'no', None],
    'charges': [16884, 1725, 4449, 21984, 42111, 48795, 3756, 28923]
}
# data
df = pd.DataFrame(data)
# print(df)

# # data information 
# print(df.shape)
# print(df.info())
# print(df.describe())


# # check nul / duplicate value
mode_children = df['children'].mode() 
# print(mode_children)
mode_sex = df['sex'].mode()
# print(mode_sex)
mode_smoker = df['smoker'].mode()
# print(mode_smoker)
mean_age = df['age'].mean() 
# print(mean_age)
mean_bmi = df['bmi'].mean() 
# print(mean_bmi)
# print(df.isna().sum())

# #  filling null value
df['children'] = df['children'].fillna(mode_children[0])
df['sex'] = df['sex'].fillna(mode_sex[0])
df['smoker'] = df['smoker'].fillna(mode_smoker[0])
df['age'] = df['age'].fillna(mean_age)
df['bmi'] = df['bmi'].fillna(mean_bmi)
print(df.isna().sum())


fig , axes = plt.subplots(2,2, figsize = (8,6))
sns.countplot(x='sex', data= df , ax= axes[0,0])
sns.countplot(x='smoker', data= df , ax= axes[0,1])
sns.countplot(x='children', data= df , ax= axes[1,0])
sns.histplot(x = 'bmi', data= df,ax=axes[1,1])
plt.show()

df['sex'] = df['sex'].map({'male' : 0 , 'female' : 1})
df['smoker'] = df['smoker'].map({'yes' : 0 , 'no' : 1})
df['children'] = df['children'].astype(int)
df['age'] = df['age'].astype(int)
print(df)

X = df[['age' , 'sex', 'bmi','children', 'smoker']]
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

#  scale data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  make model 
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
print(f'mean absolte error : {mae}')

mse = mean_squared_error(y_test,y_pred)
print(f'mean square error : {mse}')

r_2 = r2_score(y_test,y_pred)
print(f'r2_score : {r_2}')
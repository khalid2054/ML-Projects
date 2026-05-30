import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score,classification_report

# data load
df = pd.read_csv('titanic_eda')
print(df.head(10))

#  cleaning
df.dropna(subset=['embarked'],inplace=True)
df.info()

# labeling data
df['sex']= df['sex'].map({'male' : 1 , 'female' : 0})
df['alone'] = df['alone'].map({True : 1 , False : 0}) 
df = pd.get_dummies(df , columns=['embarked'], drop_first= True)

# data copy 
df_cleaned = df.copy()
final_df = df_cleaned.astype(int)
print(final_df.head())

# splits
X = df_cleaned.drop('survived',axis=1)
y = df_cleaned['survived']

#unscaled 
X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(X, y, test_size=0.20, random_state=42)

# scaled 
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)


#  logistic model
model = LogisticRegression()
model.fit(X_train_scaled, y_train_scaled)
y_predict = model.predict(X_test_scaled)

print(f'logistic model f-1 score : {f1_score(y_test_scaled,y_predict,average="weighted")}')
print(f'logistic model accuray score : {accuracy_score(y_test_scaled,y_predict)}')
print(f'logistic model calssification report : {classification_report(y_test_scaled,y_predict)}')


# model 2
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train_scaled,y_train_scaled)
knn_y_predict = knn_model.predict(X_test_scaled)

print(f'KNN model f-1 score : {f1_score(y_test_scaled,knn_y_predict,average="weighted")}')
print(f'KNN model accuray score : {accuracy_score(y_test_scaled,knn_y_predict)}')
print(f'KNN model calssification report : {classification_report(y_test_scaled,knn_y_predict)}')


# model 3
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train_unscaled,y_train_unscaled)
predict_nb = nb_model.predict(X_test_unscaled)

print(f'NB model f-1 score : {f1_score(y_test_unscaled,predict_nb,average="weighted")}')
print(f'NB model accuray score : {accuracy_score(y_test_unscaled,predict_nb)}')
print(f'NB model calssification report : {classification_report(y_test_unscaled,predict_nb)}')


# model 4
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_unscaled,y_train_unscaled)
dt_y_predict = model_dt.predict(X_test_unscaled)

print(f'DT model f-1 score : {f1_score(y_test_unscaled,dt_y_predict,average="weighted")}')
print(f'DT model accuray score : {accuracy_score(y_test_unscaled,dt_y_predict)}')
print(f'DT model calssification report : {classification_report(y_test_unscaled,dt_y_predict)}')


# model 5
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf')
svm_model.fit(X_test_scaled,y_test_scaled)
svm_y_predd = svm_model.predict(X_test_scaled)


print(f'SVM model f-1 score : {f1_score(y_test_scaled,svm_y_predd,average="weighted")}')
print(f'SVM model accuray score : {accuracy_score(y_test_scaled,svm_y_predd)}')
print(f'SVM model calssification report : {classification_report(y_test_scaled,svm_y_predd)}')
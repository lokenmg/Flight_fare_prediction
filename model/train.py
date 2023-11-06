from os import PathLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import pandas as pd
import pathlib

df = pd.read_csv(pathlib.Path('data/milk.csv'))
print(df.head())
x=df.drop("Grade_category",axis=1)
y=df["Grade_category"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


rf=RandomForestClassifier()
rf.fit(x_train,y_train)

y_pred4=rf.predict(x_test)


from sklearn.metrics import accuracy_score

print("acc rf",accuracy_score(y_test,y_pred4))
rf_model=RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
rf_model.fit(x,y)
import joblib
joblib.dump(rf_model,pathlib.Path('model/milk_prediction-v1.joblib'))
print("se guardo")
model=joblib.load("model/milk_prediction-v1.joblib")
print("se cargo")
print(x.head())
print(model.predict(x.head()))
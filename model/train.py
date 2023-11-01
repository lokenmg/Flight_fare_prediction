from os import PathLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import pathlib

df = pd.read_csv(pathlib.Path('data/Clean_Dataset.csv'))

y = df.pop('price')
columns_to_drop = ['airline','flight','source_city','departure_time','stops','arrival_time','destination_city','class']
X = df.drop(columns_to_drop, axis=1)
print (X.head())
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
print ('Training model.. ')
clf = RandomForestClassifier(n_estimators = 10,
                            max_depth=2,
                            random_state=0)
clf.fit(X_train, y_train)
#print ('Saving model..')

dump(clf, pathlib.Path('model/flight-price-v1.joblib'))

from os import PathLike
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from joblib import dump
import pandas as pd
import pathlib


data = pd.read_csv('data/Clean_Dataset2.csv', index_col=0)
#data.info()

#data.isnull().sum()

#  Remove redundant columns
dataset = data.drop(columns=["flight", "arrival_time", "departure_time"])

y = dataset['price']
X = dataset.drop(columns=['price'])

# Split the data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical columns
categorical_columns = ['airline', 'source_city', 'stops', 'destination_city', 'clase']
numerical_columns = ['duration', 'days_left']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])


# Preprocessing of training data 
X_train_transformed = preprocessor.fit_transform(X_train)

X_test_transformed = preprocessor.transform(X_test)

# Define a dictionary of models and their respective hyperparameter grids
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'param_grid': {}  # Add hyperparameters for Linear Regression if needed
    },
    'Random Forest': {
        'model': RandomForestRegressor(),
        'param_grid': {
            'n_estimators': [40, 60, 100],
            'max_depth': [10, 20, 30],
            'min_samples_split': [3, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Support Vector Machine': {
        'model': SVR(),
        'param_grid': {
            'C': [1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    }
}

X_train_subset = X_train_transformed[:10000]
y_train_subset = y_train[:10000]

best_models = {}

# Loop through each model and perform GridSearchCV
for model_name, model_info in models.items():
    model = model_info['model']
    param_grid = model_info['param_grid']
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
    
    grid_search.fit(X_train_subset, y_train_subset)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Store the best model and hyperparameters
    best_models[model_name] = {'best_params': best_params, 'best_model': best_model}

X_test_subset = X_test_transformed[:10000]
y_test_subset = y_test[:10000]

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Evaluate the best models on subset of the test data
for model_name, model_info in best_models.items():
    best_model = model_info['best_model']
    y_pred_best = best_model.predict(X_test_subset)
    
    mae_best = mean_absolute_error(y_test_subset, y_pred_best)
    mse_best = mean_squared_error(y_test_subset, y_pred_best)
    r2_best = r2_score(y_test_subset, y_pred_best)
    
    print(f"{model_name} Metrics:")
    print(f"Mean Absolute Error (MAE): {mae_best:.2f}")
    print(f"Mean Squared Error (MSE): {mse_best:.2f}")
    print(f"R-squared (R2): {r2_best:.2f}")
    print(f"The parameters of the best model:\n{model_info['best_params']}")
    print()

from sklearn.ensemble import RandomForestRegressor

# Define the best hyperparameters
best_rf_params = {
    'n_estimators': 100,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

best_rf_model = RandomForestRegressor(**best_rf_params)
best_rf_model.fit(X_train_transformed, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the testing data
y_pred = best_rf_model.predict(X_test_transformed)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

import joblib

model_filename = 'model/final_rf_model.joblib'
joblib.dump(best_rf_model, model_filename)

print(f"Model saved as '{model_filename}'")
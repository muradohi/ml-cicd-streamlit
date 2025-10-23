from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import os
import pandas as pd
from joblib import load

config_path = '/Users/murad/ml-cicd-streamlit/ml-cicd-streamlit/configs/config.yml'
def model_pipeline(num_feat, cat_feat):
    num_pipeline = Pipeline(
        [('simple_imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler())
         ]
    )

    cat_pipeline = Pipeline(
        [('simple_imputer', SimpleImputer(strategy='most_frequent')),
         ('scaler', OneHotEncoder(handle_unknown='ignore'))
         ]
    )

    preprocessor = ColumnTransformer(
        [('num_pipeline', num_pipeline, num_feat),
         ('cat_pipeline', cat_pipeline, cat_feat)
         ]
    )
    return preprocessor

def build_model(config_path, num_feat, cat_feat):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    param_grid = {
        'model__C': config['model']['param_grid']['c'],
        'model__penalty': config['model']['param_grid']['penalty'],
        'model__solver': config['model']['param_grid']['solver']
    }
    preprocessor = model_pipeline(num_feat, cat_feat)
    model = LogisticRegression(random_state=42, max_iter=1000)

    clf = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    
    grid_cv = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=config['model']['scoring'],
        cv=config['model']['cv_folds'],
        n_jobs=-1,
        verbose=2
    )

    return clf, grid_cv

def train_eval(X_train, y_train, X_test, y_test, num_feat, cat_feat):
    clf, grid_cv = build_model(config_path, num_feat, cat_feat)

    grid_cv.fit(X_train, y_train)

    best_model = grid_cv.best_estimator_
    best_params = grid_cv.best_params_
    best_score = grid_cv.best_score_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    print(f"Test Accuracy: {accuracy}")
    # confusion_matrix.plot()

    return best_params, best_score, best_model, y_pred, accuracy, confusion_matrix

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(path):
    return load(path)
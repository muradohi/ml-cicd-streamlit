import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

config_path = '/Users/murad/ml-cicd-streamlit/ml-cicd-streamlit/configs/config.yml'
def load_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    csv_path = config['data']['csv_path']
    df = pd.read_csv(csv_path)

    X = df.drop(['Churn', 'Employee_ID'], axis=1)
    y = df['Churn']

    num_feat = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feat = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, num_feat, cat_feat

# X_train, X_test, y_train, y_test, num_feat, cat_feat = load_date(config_path)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print("Numerical features:", num_feat)
# print("Categorical features:", cat_feat)

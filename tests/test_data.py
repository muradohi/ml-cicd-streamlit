from src.data import load_data

import pytest

def test_data_loading():
    X_train, X_test, y_train, y_test, num_feat, cat_feat = load_data('/Users/murad/ml-cicd-streamlit/ml-cicd-streamlit/configs/config.yml')

    assert X_train.isnull().sum() == 0, "Training features contain null values"
    assert X_test.isnull().sum() == 0, "Testing features contain null values"

    assert X_train.shape[0] > 0

    assert len(y_train.unique()) == 2, "Training target variable should be binary"
    assert len(y_test.unique()) == 2, "Testing target variable should be binary"

    
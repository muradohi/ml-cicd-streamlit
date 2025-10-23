from src.data import load_data
import pytest

def test_data_loading():
    X_train, X_test, y_train, y_test, num_feat, cat_feat = load_data(
        'configs/config.yml'
    )

    # Check no nulls
    assert X_train.isnull().sum().sum() == 0, "Training features contain null values"
    assert X_test.isnull().sum().sum() == 0, "Testing features contain null values"

    # Check non-empty
    assert X_train.shape[0] > 0, "Training set is empty"

    # Check binary target
    assert len(y_train.unique()) == 2, "Training target variable should be binary"
    assert len(y_test.unique()) == 2, "Testing target variable should be binary"

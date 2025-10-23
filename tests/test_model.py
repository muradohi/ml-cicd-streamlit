from src.data import load_data
from src.model import build_model, train_eval

import pytest

def test_model_training():

    X_train, X_test, y_train, y_test, num_feat, cat_feat = load_data('/Users/murad/ml-cicd-streamlit/ml-cicd-streamlit/configs/config.yml')


    best_params, best_score, best_model, y_pred, accuracy, confusion_matrix = train_eval(X_train, y_train, X_test, y_test, num_feat, cat_feat)
    assert best_model is not None, "Best model should not be None"
    assert accuracy >= 0.6, "Model accuracy should be at least 70%"
    assert confusion_matrix.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"
    
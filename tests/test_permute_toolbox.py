import pytest
import numpy as np
from sklearn.datasets import make_classification
import sys
sys.path.append("..")
from permute_toolbox.permute_toolbox import *
from permute_toolbox.rsfc_tools import *


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=0, random_state=42)
    features_to_permute = [[0, 1], [2, 3]]
    return X, y, features_to_permute

def test_split_data_loo(data):
    X, y, _ = data
    X_train, y_train, X_test, y_test = split_data_loo(X, y, 0)
    assert X_train.shape[0] == X.shape[0] - 1  # Ensure one sample is left out
    assert X_test.shape[0] == 1  # Ensure one sample is used for testing

def test_loocv_svm(data):
    X, y, _ = data
    predictions, weights = loocv_svm(X, y)
    assert len(predictions) == X.shape[0]
    assert len(weights) == X.shape[0]

def test_permute_single(data):
    X, y, features_to_permute = data
    permuted_row = permute_single(X[0:1], X[1:], features_to_permute[0])
    assert not np.array_equal(permuted_row, X[0:1])

def test_loocv_svm_permute_features(data):
    X, y, features_to_permute = data
    results = loocv_svm_permute_features(X, y, features_to_permute)
    assert len(results) == len(features_to_permute)
    for key, value in results.items():
        assert len(value) == X.shape[0]

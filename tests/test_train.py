import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

def test_data_loading():
    assert os.path.exists('data/dataset.csv'), "Dataset file not found"

    data = pd.read_csv('data/dataset.csv')
    assert not data.empty, "Dataset is empty"

    expected_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'target']
    assert list(data.columns) == expected_cols, "Column names do not match"

def test_model_training():
    data = pd.read_csv('data/dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    assert True
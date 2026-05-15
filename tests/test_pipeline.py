# tests/test_pipeline.py

import yaml

from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model


with open("config/params.yaml", "r") as file:
    config = yaml.safe_load(file)



def test_load_and_preprocess_data():

    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty

def test_train_and_save_model():

    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    metrics = train_and_save_model(
        X_train,
        y_train,
        X_test,
        y_test,
        config
    )

    assert isinstance(metrics, dict)

    assert "accuracy" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
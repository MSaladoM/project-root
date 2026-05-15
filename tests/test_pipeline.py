from src.model_trainer import get_model

def test_get_model_random_forest():
    config = {
        "model": {
            "name": "RandomForest",
            "n_estimators": 10,
            "max_depth": 5
        },
        "data": {"random_state": 42}
    }
    model = get_model(config)
    assert str(type(model)) == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>"
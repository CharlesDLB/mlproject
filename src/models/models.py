from typing import Callable
from abc import ABC, abstractmethod
import yaml

import pandas as pd
import numpy as np
import lightgbm


class Model(ABC):
    """Abstract base class for models."""
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def get_score(self, scorers:list[Callable], y_pred:np.ndarray|pd.Series, y_test:np.ndarray|pd.Series) -> dict[str,float]:
        """Compute the scores"""
        score_dict = {scorer.__name__: scorer(y_pred, y_test) for scorer in scorers}
        return score_dict


class LGBMRegressor(Model):
    """LGBMRegressor model."""
    def __init__(self, n_estimators: int = 20, max_depth: int = 5) -> None:
        self.model = lightgbm.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class ModelFactory:
    """Factory to create model instances."""
    @staticmethod
    def get_model(model_name: str, **params) -> Model:

        config = load_config()

        if params:
            model_class = globals()[model_name]
            return model_class(**params)
        



    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
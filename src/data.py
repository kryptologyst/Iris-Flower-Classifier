import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger
from typing import Tuple


def load_iris_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    logger.info(
        f"Iris loaded: {len(X_train)} train, {len(X_test)} test, "
        f"{len(feature_names)} features, {len(class_names)} classes"
    )
    return X_train, X_test, y_train, y_test, feature_names, class_names

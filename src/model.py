import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger
from typing import Optional
import joblib


class IrisClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        criterion: str = "gini",
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
        )
        self.feature_names_: list = []
        self.class_names_: list = []

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: list = None,
        class_names: list = None,
    ) -> "IrisClassifier":
        self.feature_names_ = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]
        self.class_names_ = class_names or [
            f"class_{c}" for c in np.unique(y)
        ]
        self.model.fit(X, y)
        logger.info(
            f"Decision Tree fitted: depth={self.model.get_depth()}, "
            f"leaves={self.model.get_n_leaves()}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "classification_report": classification_report(
                y, y_pred, target_names=self.class_names_, output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

    def get_tree_text(self) -> str:
        return export_text(
            self.model,
            feature_names=self.feature_names_,
        )

    def get_feature_importance(self) -> dict:
        return dict(
            zip(self.feature_names_, self.model.feature_importances_)
        )

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import IrisClassifier
from src.data import load_iris_data


class TestIrisClassifier:
    @pytest.fixture
    def data(self):
        return load_iris_data()

    @pytest.fixture
    def fitted_model(self, data):
        X_train, _, y_train, _, fn, cn = data
        model = IrisClassifier(max_depth=3)
        model.fit(X_train, y_train, fn, cn)
        return model

    def test_fit(self, fitted_model, data):
        _, X_test, _, y_test, _, _ = data
        preds = fitted_model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_accuracy_above_90(self, fitted_model, data):
        _, X_test, _, y_test, _, _ = data
        results = fitted_model.evaluate(X_test, y_test)
        assert results["accuracy"] > 0.90

    def test_proba_sums_to_one(self, fitted_model, data):
        _, X_test, _, _, _, _ = data
        proba = fitted_model.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importance(self, fitted_model):
        imp = fitted_model.get_feature_importance()
        assert len(imp) == 4
        assert sum(imp.values()) == pytest.approx(1.0, abs=0.01)

    def test_tree_text(self, fitted_model):
        text = fitted_model.get_tree_text()
        assert "petal" in text.lower()

    def test_save_load(self, fitted_model, tmp_path):
        p = str(tmp_path / "model.joblib")
        fitted_model.save(p)
        m2 = IrisClassifier()
        m2.load(p)
        assert m2.model is not None

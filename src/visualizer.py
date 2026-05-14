import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from loguru import logger


class IrisVisualizer:
    @staticmethod
    def plot_decision_boundaries(
        model, X: np.ndarray, y: np.ndarray, feature_names: list,
        save_path: Optional[Path] = None,
    ) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        pair_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for ax, (i, j) in zip(axes, pair_idx):
            x_min, x_max = X[:, i].min() - 0.5, X[:, i].max() + 0.5
            y_min, y_max = X[:, j].min() - 0.5, X[:, j].max() + 0.5
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100),
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            full = np.zeros((len(grid), X.shape[1]))
            full[:, i] = grid[:, 0]
            full[:, j] = grid[:, 1]
            Z = model.predict(full).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
            ax.scatter(X[:, i], X[:, j], c=y, cmap="viridis", edgecolor="k", s=30)
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Decision boundaries saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_feature_importance(
        importance: dict, save_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=(8, 5))
        names = list(importance.keys())
        values = list(importance.values())
        sns.barplot(x=values, y=names, palette="viridis")
        plt.xlabel("Importance")
        plt.title("Feature Importance — Decision Tree")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        cm: list, class_names: list, save_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()

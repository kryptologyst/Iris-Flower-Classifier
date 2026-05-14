import typer
import sys
from loguru import logger

from .config import settings
from .data import load_iris_data
from .model import IrisClassifier
from .visualizer import IrisVisualizer

app = typer.Typer(help="Iris Flower Classifier CLI")

logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def train(
    max_depth: int = typer.Option(None, help="Max tree depth"),
    visualize: bool = typer.Option(True, help="Generate plots"),
):
    logger.info("Training Decision Tree on Iris dataset...")
    X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_data()
    model = IrisClassifier(max_depth=max_depth)
    model.fit(X_train, y_train, feature_names, class_names)
    results = model.evaluate(X_test, y_test)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info("\n" + model.get_tree_text())
    logger.info(f"\nFeature Importance: {model.get_feature_importance()}")
    if visualize:
        vis = IrisVisualizer()
        vis.plot_feature_importance(
            model.get_feature_importance(),
            save_path=settings.plots_dir / "feature_importance.png",
        )
        vis.plot_confusion_matrix(
            results["confusion_matrix"], class_names,
            save_path=settings.plots_dir / "confusion_matrix.png",
        )
    model.save(str(settings.models_dir / "iris_model.joblib"))
    logger.success("Training complete!")


@app.command()
def predict(
    sepal_length: float = typer.Option(..., help="Sepal length in cm"),
    sepal_width: float = typer.Option(..., help="Sepal width in cm"),
    petal_length: float = typer.Option(..., help="Petal length in cm"),
    petal_width: float = typer.Option(..., help="Petal width in cm"),
):
    model = IrisClassifier()
    model_path = settings.models_dir / "iris_model.joblib"
    if model_path.exists():
        model.load(str(model_path))
    else:
        logger.warning("No saved model. Train first with: python -m src.main train")
        raise typer.Exit(1)
    X = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    logger.success(f"Predicted: {model.class_names_[pred]}")
    logger.info(f"Probabilities: {dict(zip(model.class_names_, proba))}")


if __name__ == "__main__":
    app()

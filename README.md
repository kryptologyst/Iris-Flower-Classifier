# Iris Flower Classifier

Decision Tree classifier on the classic **Iris dataset** — 150 samples, 4 features, 3 species.

## Overview

- Trains a configurable Decision Tree (`max_depth`, `gini`/`entropy`)
- Visualizes decision boundaries across all feature pairs
- Reports feature importance, confusion matrix, and full tree rules
- **Streamlit dashboard** for interactive training and live prediction

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main train --max-depth 3
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## Project Structure

```
0004 Iris Flower Classifier/
├── app.py              # Streamlit dashboard
├── Dockerfile / docker-compose.yml
├── src/
│   ├── model.py        # DecisionTreeClassifier wrapper
│   ├── data.py         # sklearn iris loader + train/test split
│   ├── visualizer.py   # Decision boundaries, importance, confusion matrix
│   └── main.py         # Typer CLI
└── tests/test_model.py # 6 tests
```

## License

MIT
# Iris-Flower-Classifier

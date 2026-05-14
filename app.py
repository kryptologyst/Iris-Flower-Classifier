import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_iris_data
from src.model import IrisClassifier

st.set_page_config(page_title="Iris Classifier", page_icon="", layout="wide")
st.title("Iris Flower Classifier — Decision Tree")
st.markdown("Classify iris species from sepal/petal measurements.")

X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_data()

tab1, tab2 = st.tabs(["Train & Evaluate", "Live Predict"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Max Depth", 1, 10, 3)
        criterion = st.selectbox("Criterion", ["gini", "entropy"])
    with col2:
        st.metric("Train Samples", len(X_train))
        st.metric("Test Samples", len(X_test))
        st.metric("Classes", len(class_names))

    if st.button("Train Model", type="primary"):
        with st.spinner("Training..."):
            model = IrisClassifier(max_depth=max_depth, criterion=criterion)
            model.fit(X_train, y_train, feature_names, class_names)
            results = model.evaluate(X_test, y_test)
        st.success(f"Accuracy: {results['accuracy']:.2%}")
        st.text("Decision Tree Rules:\n" + model.get_tree_text())
        imp = model.get_feature_importance()
        st.bar_chart(pd.Series(imp))

with tab2:
    st.subheader("Enter Measurements")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sl = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    with c2:
        sw = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
    with c3:
        pl = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    with c4:
        pw = st.number_input("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)

    if st.button("Classify", type="primary"):
        model = IrisClassifier(max_depth=3)
        model.fit(X_train, y_train, feature_names, class_names)
        X = np.array([[sl, sw, pl, pw]])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        st.success(f"**{class_names[pred]}**")
        st.dataframe(
            pd.DataFrame({"Species": class_names, "Probability": proba})
            .set_index("Species"),
            use_container_width=True,
        )

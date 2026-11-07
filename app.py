
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification System")

st.write(
    "A machine learning application to classify dry bean varieties "
    "using five trained classification models."
)

st.divider()

# --------------------------------------------------
# 2. MODEL FILE LOCATIONS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# --------------------------------------------------
# 3. LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        file_path = MODEL_DIR / filename

        artifact = joblib.load(file_path)

        loaded_models[model_name] = artifact

    return loaded_models


try:
    models = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.stop()

# --------------------------------------------------
# 4. UPLOAD TEST DATA
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the 16 input features and Class column",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(test_data.head())

        st.write("Dataset shape:", test_data.shape)

        # --------------------------------------------------
        # 5. VALIDATE DATASET
        # --------------------------------------------------

        first_artifact = models["Logistic Regression"]

        feature_names = first_artifact["feature_names"]

        required_columns = feature_names + ["Class"]

        missing_columns = [
            column for column in required_columns
            if column not in test_data.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Select only the required input features
        X = test_data[feature_names].copy()

        # Convert input features to numeric values
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().values.any():

            st.error(
                "The uploaded dataset contains missing or "
                "non-numeric feature values."
            )
            st.stop()

        # Get actual target values
        actual_classes = test_data["Class"].astype(str)

        label_encoder = first_artifact["label_encoder"]

        try:
            y_true = label_encoder.transform(actual_classes)

        except ValueError:

            st.error(
                "The Class column contains unknown bean categories. "
                "Please check the target labels."
            )
            st.stop()

        # --------------------------------------------------
        # 6. MODEL SELECTION
        # --------------------------------------------------

        st.divider()

        st.header("2. Select Classification Model")

        selected_model_name = st.selectbox(
            "Choose a machine learning model",
            list(MODEL_FILES.keys())
        )

        # --------------------------------------------------
        # 7. EVALUATE ALL FIVE MODELS
        # --------------------------------------------------

        st.header("3. Model Performance Comparison")

        all_results = {}
        all_predictions = {}

        for model_name, artifact in models.items():

            model = artifact["model"]
            scaler = artifact["scaler"]
            encoder = artifact["label_encoder"]

            # Apply the saved scaler
            X_scaled = scaler.transform(X)

            # Predict target classes
            predictions = model.predict(X_scaled)

            # Predict class probabilities
            probabilities = model.predict_proba(X_scaled)

            # Calculate six evaluation metrics
            accuracy = accuracy_score(y_true, predictions)

            auc = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro"
            )

            precision = precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )

            recall = recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_true,
                predictions
            )

            all_results[model_name] = {
                "Accuracy": accuracy,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "MCC": mcc
            }

            all_predictions[model_name] = predictions

        # Display comparison table
        results_df = pd.DataFrame(all_results).T

        st.dataframe(
            results_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # --------------------------------------------------
        # 8. SELECTED MODEL RESULTS
        # --------------------------------------------------

        st.divider()

        st.header("4. Selected Model Evaluation")

        selected_predictions = all_predictions[
            selected_model_name
        ]

        selected_metrics = all_results[
            selected_model_name
        ]

        st.subheader(selected_model_name)

        # Display metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{selected_metrics['Accuracy']:.4f}"
        )

        col2.metric(
            "AUC",
            f"{selected_metrics['AUC']:.4f}"
        )

        col3.metric(
            "Precision",
            f"{selected_metrics['Precision']:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{selected_metrics['Recall']:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{selected_metrics['F1 Score']:.4f}"
        )

        col6.metric(
            "MCC",
            f"{selected_metrics['MCC']:.4f}"
        )

        # --------------------------------------------------
        # 9. CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        class_names = label_encoder.classes_

        cm = confusion_matrix(
            y_true,
            selected_predictions,
            labels=np.arange(len(class_names))
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        # --------------------------------------------------
        # 10. CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            selected_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        st.success("Model evaluation completed successfully!")

    except Exception as error:

        st.error(f"An error occurred during evaluation: {error}")

else:

    st.info(
        "Please upload a test CSV file to begin model evaluation."
    )

st.divider()

st.caption(
    "Dry Bean Classification Project | Machine Learning Assignment"
)

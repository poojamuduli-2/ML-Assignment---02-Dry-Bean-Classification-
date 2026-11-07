# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify different varieties of dry beans using machine learning algorithms. Five supervised classification models are implemented and evaluated to compare their performance in identifying bean categories based on physical characteristics.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The dataset contains 13,611 records representing seven varieties of dry beans.

- **Total instances:** 13,611
- **Input features:** 16 numerical features
- **Target variable:** Class
- **Number of classes:** 7
- **Missing values:** 0

The seven bean categories are BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.

The dataset was divided into 80% training data (10,888 records) and 20% testing data (2,723 records). StandardScaler was applied to the input features, and LabelEncoder was used to encode the target categories.

## 3. Machine Learning Models Used

Five classification algorithms were implemented using Python and Scikit-learn:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Model Performance Comparison

All five models were evaluated on the same test dataset using six evaluation metrics.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| **SVM** | **0.9221** | **0.9949** | **0.9359** | **0.9330** | **0.9343** | **0.9059** |

Precision, recall, and F1 score were calculated using macro averaging. AUC was calculated using the multiclass one-vs-rest approach with macro averaging.

## 5. Observations

### Logistic Regression
Logistic Regression achieved 92.14% accuracy and an AUC of 0.9948. It demonstrated strong and balanced classification performance and produced results very close to SVM.

### Decision Tree
Decision Tree achieved 89.20% accuracy, the lowest among the five models. Although it classified most test samples correctly, its overall performance was lower than the other classifiers.

### K-Nearest Neighbors
KNN achieved 91.66% accuracy and an F1 score of 0.9293. It performed well and was the third-best model based on accuracy.

### Gaussian Naive Bayes
Gaussian Naive Bayes achieved 89.79% accuracy and an AUC of 0.9916. Its high AUC indicates strong class discrimination despite its comparatively lower classification accuracy.

### Support Vector Machine
SVM achieved the highest accuracy of 92.21%, along with the highest AUC, precision, recall, F1 score, and MCC. It was the best-performing model in this experiment.

## 6. Overall Conclusion

Support Vector Machine (SVM) was identified as the best-performing classifier for the Dry Bean dataset, achieving 92.21% accuracy and an F1 score of 0.9343.

However, Logistic Regression achieved nearly identical results, suggesting that both algorithms are suitable for this classification task.

## 7. Streamlit Application

An interactive Streamlit application was developed to evaluate the five saved machine learning models.

The application supports:

- Uploading test data in CSV format.
- Selecting one of five trained classifiers.
- Displaying Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Comparing all five classifiers on the uploaded test dataset.
- Visualizing a confusion matrix.
- Viewing a classification report.

The application loads previously trained models and does not retrain them during evaluation.

**Live Streamlit App:** https://dry-bean-classification-2026ak05702.streamlit.app/

## 8. GitHub Repository

https://github.com/poojamuduli-2/ML-Assignment---02-Dry-Bean-Classification-

## 9. Project Files

- `app.py` — Streamlit application source code.
- `requirements.txt` — Required Python libraries.
- `test_data.csv` — Test dataset with input features and target labels.
- `model_comparison.csv` — Evaluation results for all five models.
- `models/` — Five saved trained classifiers and preprocessing artifacts.
- `2026AK05702_ML_Assignment_2_Dry_Bean_Classification.ipynb` — Google Colab notebook containing training and evaluation code.

## 10. Technologies Used

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib, Google Colab, GitHub, and Streamlit.

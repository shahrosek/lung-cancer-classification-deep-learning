# Lung Cancer Classification with a Deep Neural Network

> **Note:** The notebook in this repository is a clean version with all cell outputs cleared for fast rendering on GitHub.
>
> **➡️ [View the fully rendered notebook on NBViewer](https://nbviewer.org/github/shahrosek/lung-cancer-classification-deep-learning/blob/main/lung-cancer-prediction.ipynb)**
>
> **➡️ [View the original notebook on Kaggle](https://www.kaggle.com/code/shahrosek/lung-cancer-prediction)**

This repository details a binary classification project aimed at predicting the likelihood of lung cancer in patients based on tabular data. The project places a strong emphasis on addressing the critical challenge of working with a highly imbalanced dataset.

## 🎯 Objective
To build and evaluate a high-performance classifier that can accurately predict whether a patient has cancer, with a focus on maximizing the F1-Score to ensure reliability for the minority (cancer) class.

## 📊 Dataset
The dataset is a tabular collection of patient information, including features like age, gender, race, and smoking habits. The key challenge is its severe class imbalance, with the non-cancer class representing over 96% of the data.

## ⚙️ Methodology & Technical Walkthrough

1.  **Data Preprocessing**:
    * Categorical features (`gender`, `race`, `smoker`) were converted into numerical format using **One-Hot Encoding**.
    * Numerical features (`age`, `days_cancer`) were scaled using `MinMaxScaler` (Normalization) after analyzing their distribution.

2.  **Handling Class Imbalance**:
    * This was the central challenge. The following techniques from the `imbalanced-learn` library were implemented and compared:
        * **Under-sampling (NearMiss)**: Reduced the majority class to balance the dataset.
        * **Over-sampling (RandomOverSampler)**: Increased the minority class by duplicating existing samples.
        * **SMOTE (Synthetic Minority Over-sampling Technique)**: Generated new, synthetic samples for the minority class to create a perfectly balanced dataset of over 100,000 samples. This proved to be the most effective method.

3.  **Model Implementation & Evaluation**:
    * **Traditional Models**: `RandomForestClassifier`, `CatBoostClassifier`, `DecisionTreeClassifier`, and `LogisticRegression` were trained on the balanced data.
    * **Deep Neural Network**: A sequential model was built using **Keras** with multiple dense layers and `ReLU` activation, and a final `Sigmoid` activation layer for binary classification.

## 📈 Results & Outcome
After balancing the data with **SMOTE**, the models achieved exceptional performance, highlighting the effectiveness of the technique.

* **Deep Neural Network F1-Score:** **0.9998**
* **Random Forest / CatBoost F1-Score:** **1.0** (Perfect Score)

The results demonstrate that by synthetically balancing the dataset, machine learning models can achieve extremely high accuracy in identifying lung cancer cases from the given features.

## 🛠️ Tech Stack
* **Language**: `Python`
* **Libraries**: `Pandas`, `NumPy`, `Scikit-learn`, `imblearn` (SMOTE, NearMiss), `CatBoost`, `TensorFlow`, `Keras`.

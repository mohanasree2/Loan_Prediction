# Loan Approval Prediction System
This project implements a robust machine learning pipeline designed to predict loan approval status based on applicant data. The system utilizes advanced data preprocessing, feature selection techniques, and hyperparameter optimization to achieve high predictive accuracy and model reliability.

## Table of Contents
Project Description
Dataset Overview
Installation and Requirements
Machine Learning Pipeline
Model Evaluation and Results
Usage
Model Persistence

## Project Description
The primary objective of this project is to automate the loan eligibility process. By analyzing historical loan data, the model identifies key patterns and features that contribute to a successful loan application. The project incorporates a full scikit-learn pipeline to ensure data consistency and prevent data leakage during training and validation.

## Dataset Overview
The project uses the loan_approved.csv dataset. The features include demographic, financial, and credit-related information of applicants.

Target Variable: Loan_Status (Approved) (Binary classification)
Key Features: Personal information, income, loan amount, and credit history.
Data Preprocessing: Handling of missing values, scaling of numerical features, and encoding of categorical variables.

## Installation and Requirements
To run the project, ensure you have Python 3.8+ installed. The following libraries are required:
numpy
pandas
scikit-learn (version 1.6.1 or later)
pickle

## Install the dependencies using pip:

pip install numpy pandas scikit-learn

## Machine Learning Pipeline
The project implements a comprehensive Pipeline that includes:

1. Data Preprocessing
- Numerical Pipeline: Imputation of missing values using the median strategy followed by standard scaling (StandardScaler).
- Categorical Pipeline: Imputation of missing values using the most frequent strategy followed by one-hot encoding (OneHotEncoder).
- ColumnTransformer: Combines both pipelines to handle heterogeneous data types simultaneously.

2. Feature Selection
Utilizes SelectKBest with the f_classif (ANOVA F-value) scoring function to select the most significant features, reducing model complexity and preventing overfitting.

3. Hyperparameter Tuning
GridSearchCV was employed to perform an exhaustive search over a specified parameter grid.
Models evaluated include LogisticRegression and RandomForestClassifier.

## Model Evaluation and Results
The final model was selected based on its performance during cross-validation.

### Best Model Performance
- Algorithm: Logistic Regression
- Hyperparameters: C=0.1, k=10 features selected.
- Train Accuracy: 79.78%
- Test Accuracy: 84.42%

### Cross-Validation Results (5-Fold)
- Mean Train Accuracy: 81.23%
- Mean Test Accuracy: 80.95%
- Test Accuracy Variance: 0.000459

The results indicate a stable model with high generalization capability and low variance across different data splits.

### Usage
- The main workflow is contained within the Feature_selection.ipynb notebook. To replicate the results:
- Load the dataset loan_approved.csv.
- Execute the preprocessing cells to clean and transform the data.
- Run the GridSearchCV cell to identify the optimal parameters.
- Evaluate the model using the provided test set and cross-validation scripts.

Model Persistence
The final trained model is serialized and saved as model.pkl using the pickle library. This allows for easy deployment and future inference without needing to retrain the pipeline.

stremlit app link : https://loanprediction-n6xho7icqobeiibklkzx7i.streamlit.app/

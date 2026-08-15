# SMS & Email Spam Classifier

An end-to-end Natural Language Processing (NLP) and classical Machine Learning project for classifying SMS and email messages as **spam** or **ham** using text preprocessing, TF-IDF feature extraction, comparative model evaluation, and a Streamlit inference application.

## Overview

Spam detection is a binary text-classification problem where the model must distinguish unwanted messages from legitimate communication.

This project implements the complete machine learning workflow:

- Dataset cleaning and duplicate removal
- Exploratory data analysis
- NLP text preprocessing
- Tokenization, stopword removal, and stemming
- TF-IDF feature extraction
- Comparative evaluation of multiple ML classifiers
- Ensemble model experimentation
- Multinomial Naive Bayes inference
- Model serialization using Pickle
- Streamlit-based interactive prediction
- Reproducible model artifact generation

## Dataset

The project uses the **SMS Spam Collection** dataset containing **5,572 messages**.

Each message belongs to one of two classes:

- `ham` — legitimate message
- `spam` — unwanted message

The original dataset contains additional unnamed columns, which are removed during preprocessing. Duplicate messages are also removed before model training.

## Machine Learning Pipeline

```text
SMS Spam Collection
        │
        ▼
Data Cleaning
        │
        ├── Remove unused columns
        ├── Rename target/text columns
        ├── Label encode classes
        └── Remove duplicate records
        │
        ▼
Exploratory Data Analysis
        │
        ├── Class distribution
        ├── Message length analysis
        ├── Word statistics
        └── Spam/Ham corpus analysis
        │
        ▼
NLP Preprocessing
        │
        ├── Lowercasing
        ├── Tokenization
        ├── Alphanumeric filtering
        ├── Stopword removal
        └── Porter stemming
        │
        ▼
TF-IDF Feature Extraction
        │
        └── max_features = 3000
        │
        ▼
80 / 20 Train-Test Split
        │
        ▼
Model Training & Comparison
        │
        ├── SVC
        ├── KNN
        ├── Multinomial Naive Bayes
        ├── Decision Tree
        ├── Logistic Regression
        ├── Random Forest
        ├── AdaBoost
        ├── Bagging
        ├── Extra Trees
        ├── Gradient Boosting
        └── XGBoost
        │
        ▼
Model Evaluation
        │
        ├── Accuracy
        └── Precision
        │
        ▼
Serialized Inference Artifacts
        │
        ├── vectorizer.pkl
        └── model.pkl
        │
        ▼
Streamlit Application
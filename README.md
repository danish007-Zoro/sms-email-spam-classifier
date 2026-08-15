# SMS & Email Spam Classifier

An end-to-end Natural Language Processing (NLP) project that classifies SMS and email messages as **spam** or **ham** using text preprocessing, TF-IDF feature extraction, classical machine learning models, and a Streamlit web application.

## Overview

Spam messages are often short, noisy, and highly variable in language, making text classification a useful NLP problem.

This project builds a complete machine learning pipeline from raw message data to an interactive prediction application.

The workflow includes:

- Data cleaning and duplicate removal
- Exploratory data analysis
- NLP text preprocessing
- TF-IDF feature extraction
- Comparative evaluation of multiple machine learning classifiers
- Ensemble model experimentation
- Model serialization for inference
- Streamlit-based interactive deployment

## Dataset

The project uses the **SMS Spam Collection** dataset containing **5,572 messages**.

Each message is classified into one of two categories:

- `ham` — legitimate message
- `spam` — unwanted message

The dataset initially contains 5,572 records. Duplicate messages are removed during preprocessing before model training.

## Machine Learning Pipeline

```text
Raw SMS Dataset
       │
       ▼
Data Cleaning
       │
       ├── Remove unused columns
       ├── Encode target labels
       └── Remove duplicate messages
       │
       ▼
Exploratory Data Analysis
       │
       ├── Class distribution
       ├── Character / word / sentence statistics
       └── Message-level analysis
       │
       ▼
NLP Preprocessing
       │
       ├── Lowercasing
       ├── Tokenization
       ├── Non-alphanumeric filtering
       ├── Stopword removal
       └── Porter stemming
       │
       ▼
TF-IDF Vectorization
       │
       └── Maximum 3,000 features
       │
       ▼
Train / Test Split
       │
       └── 80 / 20 split
       │
       ▼
Model Comparison
       │
       ├── Naive Bayes
       ├── Logistic Regression
       ├── SVM
       ├── Random Forest
       ├── Extra Trees
       ├── AdaBoost
       ├── Gradient Boosting
       ├── XGBoost
       └── Other classifiers
       │
       ▼
Model Selection & Serialization
       │
       ▼
Streamlit Application
       │
       ▼
Spam / Not Spam Prediction
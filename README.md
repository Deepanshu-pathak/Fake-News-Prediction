# 📰 Fake News Prediction System

A machine learning-based system to classify news articles as **FAKE** or **REAL** using Natural Language Processing (NLP) techniques and Logistic Regression.

---

## 📌 Overview

This project leverages text classification methods to identify fake news using a combination of data preprocessing, feature extraction with TF-IDF, and model training using Logistic Regression. It is built using Python and widely used machine learning and NLP libraries.

---

## 🧰 Libraries Used

- **pandas** – for data loading and manipulation  
- **nltk** – for natural language processing tasks like stopword removal and stemming  
- **sklearn (scikit-learn)** – for vectorization, model training, and evaluation  
- **string** – for punctuation removal  
- **TfidfVectorizer** – for converting text data into numerical feature vectors  
- **LogisticRegression** – the classification algorithm used

---

## 🔄 Process Flow

### 1. Data Loading
- Load the dataset using `pandas`.
- Merge relevant fields such as `title` and `text` to form a complete news content.

### 2. Text Preprocessing
- Convert all text to lowercase for uniformity.
- Remove punctuation using Python's `string` module.
- Tokenize the text.
- Remove stopwords using NLTK’s stopword list.
- Apply stemming using NLTK's `PorterStemmer` to reduce words to their root forms.

### 3. Vectorization
- Use `TfidfVectorizer` from `sklearn.feature_extraction.text` to transform the cleaned text into numerical vectors.

### 4. Model Training
- Split the dataset into training and testing sets using `train_test_split`.
- Train a `LogisticRegression` model using the TF-IDF vectors.
- Fit the model with training data.

### 5. Model Evaluation
- Predict using the trained model on the test set.
- Evaluate performance using metrics such as:
  - Accuracy Score

---

## ✅ Key Features

- Cleans and preprocesses raw text data  
- Converts textual data to numerical form using TF-IDF  
- Trains a supervised ML model to predict news authenticity  
- Provides evaluation metrics for model assessment

---

## 📈 Results

The system achieves reliable accuracy on the test data and correctly classifies most instances of fake and real news, proving its effectiveness as a basic fake news detector.

---

## 📄 License

This project is open-source and free to use under the MIT License.

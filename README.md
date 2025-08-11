
# IMDB Movie Review Sentiment Analysis — Advanced NLP Pipeline

This project demonstrates an **end-to-end Natural Language Processing (NLP) pipeline** for sentiment analysis on the [IMDB 50k Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/).
It combines **advanced text analysis**, **custom preprocessing**, and **modern machine learning techniques** to achieve **\~0.90 accuracy** in binary sentiment classification (positive / negative).

---

## 📌 Project Overview

The workflow includes:

1. **Exploratory Data Analysis (EDA)**

   * Dataset structure & balance check
   * Word & n-gram frequency analysis
   * Vocabulary size & rare word analysis
   * Sentiment vs. review length distribution
2. **Advanced Text Preprocessing**

   * Unicode normalization
   * Smart quotes & dash standardization
   * Contraction expansion
   * Emoji-to-text conversion
   * Repeated character normalization
   * HTML entity decoding
3. **Feature Engineering**

   * TF-IDF vectorization with n-grams
   * Optional transformer embeddings for comparison
4. **Model Training & Evaluation**

   * Logistic Regression baseline
   * Evaluation with accuracy, F1-score, ROC-AUC
   * Error analysis for model improvement

---

## 📊 Dataset

* **Name:** IMDB Dataset of 50K Movie Reviews
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/)
* **Structure:**

  * `review` — movie review text
  * `sentiment` — `positive` or `negative` label

---

## ⚙️ Installation & Setup

1. Clone the repository and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Download the dataset from Kaggle and place it in the working directory.
3. Open the notebook and run all cells in order.

---

## 🚀 Results

* **Final Accuracy:** \~0.90 on the test set
* **Model:** TF-IDF + Logistic Regression (baseline)
* Higher performance possible with transformer fine-tuning (RoBERTa / DeBERTa).

---

## 📈 Future Improvements

* Fine-tune a transformer model for potentially >0.95 accuracy
* Apply data augmentation (backtranslation, paraphrasing)
* Deploy model as an API or interactive web app

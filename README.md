# AI/ML Foundation Program

This repository contains my implementations for the AI/ML Foundation Program.

The foundation phase is designed to build the core machine learning skills required before moving into the final project.

## Foundation Roadmap

| Week | Topic | Build | Status |
|------|-------|-------|--------|
| Week 1 | Pandas, Features & Labels, Data Cleaning | Load and clean a sample dataset | Completed |
| Week 2 | Linear Regression, Train/Test Split, Evaluation Metrics | Train a regression model | Completed |
| Week 3 | Text Preprocessing, TF-IDF, Naive Bayes | Build a spam classifier | Completed |

---

# Week 1 - Data Loading & Cleaning

## Objective

The Week 1 implementation focuses on:

- Pandas fundamentals
- Understanding features and labels
- Loading a sample dataset
- Identifying missing values
- Cleaning the dataset
- Separating features and labels
- Generating a cleaning report

## Week 1 Dataset

The dataset contains plant health observations with information such as:

- Plant ID
- Temperature
- Humidity
- Soil Moisture
- Leaf Color
- Leaf Area
- Disease
- Observation Date

The `Disease` column is treated as the label, while the remaining relevant columns are used as features.

## Week 1 Structure

```text
data/
├── raw/
└── processed/

src/
├── main.py
├── data_loader.py
├── data_cleaner.py
├── feature_label.py
└── report_generator.py

reports/
notebooks/
outputs/
---

# Week 3 - Text Preprocessing, TF-IDF & Naive Bayes Spam Classifier

## Objective

The Week 3 implementation focuses on:

- Cleaning and normalizing raw SMS text (lowercasing, removing URLs/punctuation, collapsing whitespace)
- Handling missing values and duplicate rows before training
- Converting text into TF-IDF (Term Frequency - Inverse Document Frequency) features
- Training a Multinomial Naive Bayes classifier for spam detection
- Splitting data into train/test sets
- Evaluating the classifier with accuracy, precision, recall, F1-score and a confusion matrix

## Week 3 Dataset

`week3/data/raw/sms_spam_dataset.csv` is a small, self-contained SMS dataset created for this
exercise so the project stays reproducible without any external downloads. Each row contains:

- `message` - the raw SMS text
- `label` - `ham` (legitimate) or `spam`

The raw file intentionally includes a few missing values, blank messages, and duplicate rows so
that the missing-value handling and deduplication steps in the pipeline have something real to do.

## Week 3 Preprocessing

- Missing/blank messages and missing/invalid labels are dropped
- Duplicate rows are removed
- Message text is lowercased, URLs are stripped, punctuation/digits are removed, and extra
  whitespace is collapsed

## Week 3 TF-IDF & Naive Bayes

- `TfidfVectorizer` (scikit-learn) converts the cleaned messages into TF-IDF feature vectors,
  with English stop words removed
- `MultinomialNB` (scikit-learn) is trained on the TF-IDF features to classify messages as
  `ham` or `spam`
- The vectorizer and classifier are combined into a single scikit-learn `Pipeline` so the exact
  same TF-IDF vocabulary is reused automatically at prediction time

## Week 3 Evaluation Metrics

The classifier is evaluated on a held-out 20% test split using:

- Accuracy
- Precision (spam)
- Recall (spam)
- F1-score (spam)
- Confusion matrix

On the current dataset the model reaches roughly 90% accuracy, 0.90 precision, 0.90 recall and
a 0.90 F1-score on the test set (exact numbers can be seen in
`week3/reports/classification_report.txt` after running the pipeline).

## Week 3 Structure

```text
week3/
data/
    raw/
    processed/

src/
    main.py
    data_loader.py
    text_preprocessor.py
    model_trainer.py
    model_evaluator.py

models/
reports/
outputs/
```

## How to Run Week 3

Run from the repository root:

```bash
pip install -r requirements.txt
python week3/src/main.py
```

This will:

1. Load and clean `week3/data/raw/sms_spam_dataset.csv`
2. Preprocess the message text
3. Split the data into train/test sets
4. Train the TF-IDF + Naive Bayes pipeline
5. Evaluate the model and print the results to the console
6. Save the trained model to `week3/models/spam_classifier.pkl`
7. Save test predictions to `week3/outputs/predictions.csv`
8. Save the evaluation report to `week3/reports/classification_report.txt`

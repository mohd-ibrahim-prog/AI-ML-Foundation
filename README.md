# AI/ML Foundation Program

A practical AI/ML foundation project covering the complete machine learning workflow from data preparation to deep learning.

This repository contains four progressive implementations developed during the AI/ML Foundation Program. Each week focuses on a different stage of the machine learning pipeline, starting with structured data and ending with image classification using a Convolutional Neural Network.

---

## Foundation Journey

| Week | Focus Area | Project |
|------|------------|---------|
| Week 1 | Data Preparation | Plant Health Data Cleaning |
| Week 2 | Machine Learning | Plant Growth Prediction |
| Week 3 | Natural Language Processing | SMS Spam Classification |
| Week 4 | Deep Learning & Computer Vision | Plant Disease Detection |

---

# Week 1 — Data Loading & Cleaning

### Focus

The first week establishes the fundamentals required for working with machine learning datasets.

The implementation covers:

- Loading datasets using Pandas
- Understanding rows, columns and data types
- Identifying missing values
- Cleaning inconsistent data
- Handling duplicate records
- Understanding features and labels
- Preparing processed datasets
- Generating data-cleaning reports

### Dataset

A plant health dataset containing observations such as:

- Plant ID
- Temperature
- Humidity
- Soil Moisture
- Leaf Color
- Leaf Area
- Disease
- Observation Date

The `Disease` column is treated as the target label, while the remaining relevant attributes are used as input features.

### Outcome

The raw dataset is transformed into a clean and structured dataset suitable for machine learning.

---

# Week 2 — Linear Regression

### Focus

Week 2 introduces supervised machine learning through a regression problem.

The implementation covers:

- Loading a cleaned dataset
- Selecting features and target variables
- Train/test splitting
- Training a Linear Regression model
- Making predictions
- Calculating evaluation metrics
- Saving the trained model
- Generating prediction outputs and a model report

### Model

A Linear Regression model is used to learn the relationship between plant-related input features and the target variable.

### Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Outcome

This week demonstrates the complete basic machine learning workflow:

**Data → Training → Prediction → Evaluation**

---

# Week 3 — Text Classification with TF-IDF & Naive Bayes

### Focus

Week 3 introduces Natural Language Processing (NLP) and text classification.

The project builds an SMS spam classifier capable of distinguishing between legitimate (`ham`) and unwanted (`spam`) messages.

### Text Processing

The implementation performs:

- Missing-value handling
- Duplicate removal
- Text normalization
- Lowercasing
- URL removal
- Punctuation and digit removal
- Whitespace normalization

### Feature Extraction

Text messages are converted into numerical machine learning features using:

**TF-IDF — Term Frequency-Inverse Document Frequency**

This allows the machine learning model to work with textual data.

### Machine Learning Model

A **Multinomial Naive Bayes** classifier is trained using the generated TF-IDF features.

The TF-IDF vectorizer and classifier are combined into a single machine learning pipeline.

### Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Outcome

This week demonstrates how raw human language can be transformed into numerical features and used to build a classification model.

---

# Week 4 — Plant Disease Detection Using CNN

### Focus

Week 4 introduces Deep Learning and Computer Vision.

The project builds a Convolutional Neural Network (CNN) capable of classifying plant leaf images into different plant disease categories.

### Dataset

The project uses the **PlantVillage Dataset**.

The dataset contains color images organized into disease/plant classes.

The current dataset contains:

- **38 disease/health classes**
- Plant leaf images organized by class
- Color images used for CNN training

The downloaded dataset is stored locally inside:

```text
week4/data/raw/PlantVillage/
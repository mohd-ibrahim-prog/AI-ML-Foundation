# AI/ML Foundation Program

A practical AI/ML foundation project covering the complete machine learning workflow from data preparation and traditional machine learning to Natural Language Processing, Deep Learning, Computer Vision, model evaluation, per-class performance analysis, and deployment of a trained AI model as a mobile-friendly web application.

This repository contains eight progressive implementations developed during the AI/ML Foundation Program. Each week builds on the previous concepts and gradually moves from basic data preparation to machine learning, NLP, deep learning, computer vision, model evaluation, and finally deployment of the trained plant disease classifier as a web application.

---

# Foundation Journey

| Week | Focus Area | Project |
|------|------------|---------|
| Week 1 | Data Preparation | Plant Health Data Cleaning |
| Week 2 | Machine Learning | Plant Growth Prediction |
| Week 3 | Natural Language Processing | SMS Spam Classification |
| Week 4 | Deep Learning & Computer Vision | Plant Disease Detection |
| Week 5 | Image Model Training & Validation | Plant Disease CNN — Training & Validation |
| Week 6 | Model Evaluation & Improvement | Plant Disease CNN — Test Evaluation & Improvement |
| Week 7 | Per-Class Evaluation | Plant Disease CNN — Per-Class Accuracy |
| Week 8 | Deployment | Mobile-Friendly Plant Disease Detection Web App |

---

# Week 1 — Data Loading & Cleaning

## Focus

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

## Dataset

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

## Outcome

The raw dataset is transformed into a clean and structured dataset suitable for machine learning.

---

# Week 2 — Linear Regression

## Focus

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

## Model

A Linear Regression model is used to learn the relationship between plant-related input features and the target variable.

## Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Outcome

This week demonstrates the complete basic machine learning workflow:

**Data → Training → Prediction → Evaluation**

---

# Week 3 — Text Classification with TF-IDF & Naive Bayes

## Focus

Week 3 introduces Natural Language Processing (NLP) and text classification.

The project builds an SMS spam classifier capable of distinguishing between legitimate (`ham`) and unwanted (`spam`) messages.

## Text Processing

The implementation performs:

- Missing-value handling
- Duplicate removal
- Text normalization
- Lowercasing
- URL removal
- Punctuation and digit removal
- Whitespace normalization

## Feature Extraction

Text messages are converted into numerical machine learning features using:

**TF-IDF — Term Frequency-Inverse Document Frequency**

This allows the machine learning model to work with textual data.

## Machine Learning Model

A **Multinomial Naive Bayes** classifier is trained using the generated TF-IDF features.

The TF-IDF vectorizer and classifier are combined into a single machine learning pipeline.

## Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Outcome

This week demonstrates how raw human language can be transformed into numerical features and used to build a text classification model.

---

# Week 4 — Plant Disease Detection Using CNN

## Focus

Week 4 introduces Deep Learning and Computer Vision.

The project builds a Convolutional Neural Network (CNN) capable of classifying plant leaf images into different plant disease categories.

## Dataset

The project uses the **PlantVillage Dataset**.

The dataset contains color images organized into disease/plant classes.

The current dataset contains:

- 38 disease/health classes
- Plant leaf images organized by class
- Color images used for CNN training

The downloaded dataset is stored locally and is not committed to the repository because of its size.

## CNN Model

The CNN uses multiple convolutional layers to learn visual features from plant leaf images.

The model includes:

- Convolutional layers
- Pooling layers
- Dropout
- Dense layers
- Softmax output for multi-class classification

## Outcome

Week 4 establishes the basic image classification pipeline:

**Images → Preprocessing → CNN Training → Disease Prediction**

The trained CNN model is saved for later use.

---

# Week 5 — Training & Validation for Images

## Focus

Week 5 extends the Week 4 plant disease detector by introducing a dedicated **training and validation workflow** for image classification.

This week focuses on monitoring model performance during training rather than relying only on the training accuracy.

The implementation includes:

- Stratified train/validation split
- Training the CNN using the training data
- Validation after every training epoch
- Training and validation accuracy tracking
- Training and validation loss tracking
- Saving training history
- Generating accuracy/loss curves
- Generating a training report

## Model

Week 5 intentionally reuses the CNN architecture introduced in Week 4.

The purpose of this week is to understand the difference between:

**Training Performance vs Validation Performance**

Model improvements such as data augmentation and early stopping are introduced in Week 6.

## Dataset

The PlantVillage dataset is reused from Week 4.

The dataset is not duplicated inside Week 5.

The pipeline can locate the dataset through:

1. `PLANTVILLAGE_DATA_DIR`
2. `week5/data/raw/PlantVillage/`
3. `week4/data/raw/PlantVillage/`

## Outputs

After training, the workflow can generate:

- Trained Week 5 CNN model
- Training history CSV
- Accuracy/loss plot
- Week 5 training report

## Outcome

Week 5 demonstrates how validation data can be used to monitor a deep learning model during training.

The workflow becomes:

**Training Data → CNN Training → Validation → Training History → Performance Analysis**

---

# Week 6 — Model Evaluation & Improvement

## Focus

Week 6 continues from Week 5 and introduces a proper **held-out test set** along with standard model improvement techniques.

The implementation includes:

- Stratified train/validation/test split
- 70% training data
- 15% validation data
- 15% test data
- Data augmentation during training
- Random image flipping
- Random rotation
- Random zoom
- Additional dropout
- Early stopping
- Model checkpointing
- Evaluation on unseen test data

## Model Improvement

The CNN from the earlier weeks is improved using beginner-friendly techniques designed to reduce overfitting.

### Data Augmentation

Data augmentation creates variations of training images using transformations such as:

- Random flipping
- Random rotation
- Random zoom

These transformations are applied during training only.

### Early Stopping

Training can stop automatically when validation performance stops improving.

The best validation weights are restored.

### Model Checkpointing

The best-performing model based on validation performance is saved.

## Evaluation

The final model is evaluated using the held-out test set.

Evaluation includes:

- Test loss
- Test accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

The test set is not used during model training.

## Recorded Result

The Week 6 model achieved a recorded test accuracy of approximately **92.89%** on the held-out PlantVillage test set used by the project.

This result applies to the project dataset and evaluation setup. Real-world photographs may produce different results because of differences in lighting, backgrounds, image quality, and other conditions.

## Outcome

Week 6 demonstrates the difference between training, validation, and testing.

The workflow becomes:

**Train → Validate → Improve → Select Best Model → Test on Unseen Data**

---

# Week 7 — Per-Class Accuracy Evaluation

## Focus

Week 7 performs a detailed evaluation of the model **for every individual plant disease/class**.

Instead of reporting only one overall accuracy value, the model's performance is analyzed separately for each class.

## Evaluation Process

Week 7:

- Loads the trained Week 6 model
- Does not retrain the model
- Rebuilds the same test split used in Week 6
- Runs predictions on the held-out test set
- Calculates metrics for every disease/class
- Identifies the strongest-performing class
- Identifies the weakest-performing class

## Per-Class Metrics

The following metrics are calculated for every class:

- Per-class accuracy
- Precision
- Recall
- F1-score
- Support

Per-class accuracy is calculated as the proportion of images belonging to that class that were correctly classified.

For a class, this is numerically equivalent to its recall.

## Outputs

The evaluation generates:

- Per-class metrics CSV
- Classification report
- Confusion matrix
- Week 7 summary
- Strongest and weakest class analysis

## Relationship with Week 6

Week 7 uses the model produced by Week 6.

Therefore:

**Week 6 → Train & Select Best Model**

**Week 7 → Analyze That Model Per Class**

Week 7 must be run after Week 6.

## Outcome

Week 7 provides a more detailed understanding of model performance and helps identify which plant disease classes are easier or harder for the CNN to classify.

---

# Week 8 — Mobile-Friendly Deployment

## Focus

Week 8 converts the trained plant disease classification model into a simple **mobile-friendly web application**.

The goal is to make the trained AI model accessible through a browser instead of requiring the user to run Python commands directly.

The application allows a user to:

1. Select or take a plant leaf photograph.
2. Upload the image.
3. Send the image to the trained CNN.
4. Receive the predicted plant disease class.
5. View the confidence percentage.
6. View the top three predicted classes.

The model is **not retrained in Week 8**.

The application uses the trained model produced and evaluated in Week 6.

---

## Web Application

The Week 8 application is built using **Flask**.

The application contains:

- Flask backend
- HTML interface
- CSS styling
- JavaScript for browser interaction
- Image upload and preview
- Image validation
- CNN prediction
- Confidence display
- Top-3 predictions
- Error handling
- Health-check endpoint

The application is designed to be usable from a desktop browser as well as a mobile browser.

---

## Model Used

The deployed application uses:

`week6/models/plant_disease_cnn_week6.keras`

The Week 6 model is used because it is the improved CNN model that was evaluated on the held-out test set and subsequently used by Week 7 for per-class evaluation.

The deployment therefore follows:

**Week 6 → Trained & Evaluated Model**

**Week 7 → Detailed Model Analysis**

**Week 8 → Deployment of the Evaluated Model**

---

## Prediction Workflow

The complete Week 8 prediction workflow is:

```text
User selects/takes leaf image
            ↓
Image uploaded through browser
            ↓
File validation
            ↓
Image opened and verified
            ↓
RGB conversion
            ↓
Resize to 128 × 128
            ↓
Convert to float32
            ↓
Pixel values divided by 255
            ↓
CNN model prediction
            ↓
Softmax probabilities
            ↓
Top prediction + Top 3 predictions
            ↓
Confidence displayed in web interface
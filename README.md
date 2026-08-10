# AI/ML Foundation Program

This repository contains my implementations for the AI/ML Foundation Program.

The foundation phase is designed to build the core machine learning skills required before moving into the final project.

## Foundation Roadmap

| Week | Topic | Build | Status |
|------|-------|-------|--------|
| Week 1 | Pandas, Features & Labels, Data Cleaning | Load and clean a sample dataset | Completed |
| Week 2 | Linear Regression, Train/Test Split, Evaluation Metrics | Train a regression model | Completed |
| Week 3 | Text Preprocessing, TF-IDF, Naive Bayes | Build a spam classifier | Upcoming |

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
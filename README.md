# AI-Powered Plant Disease Detection & Analysis System

> A CNN-based plant leaf analysis system with a validation and uncertainty-aware decision layer. It can **accept** a prediction, or **withhold** it and explain why.

![Phase](https://img.shields.io/badge/Phase-Capstone%20Scope%20Finalization-blue)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/Model-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-2E8B57)
![Deployment](https://img.shields.io/badge/Deployment-Planned%3A%20Render-lightgrey)

---

## Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [How the System Works](#4-how-the-system-works)
5. [Dataset and Supported Classes](#5-dataset-and-supported-classes)
6. [CNN Model and Training](#6-cnn-model-and-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Reliability and Decision Layer](#8-reliability-and-decision-layer)
9. [Web Application](#9-web-application)
10. [Project Structure](#10-project-structure)
11. [Capstone Scope, Prototype and Deployment](#11-capstone-scope-prototype-and-deployment)
12. [Demonstration and Testing Plan](#12-demonstration-and-testing-plan)
13. [Limitations and Responsible Use](#13-limitations-and-responsible-use)
14. [Skills Used](#14-skills-used)
15. [Future Scope](#15-future-scope)
16. [Roadmap and Current Status](#16-roadmap-and-current-status)
17. [Conclusion](#17-conclusion)
18. [Disclaimer](#18-disclaimer)

---

## Status Legend

| Tag | Meaning |
|---|---|
| **CURRENT** | Developed in the earlier AI/ML work and carried into this repository as the project basis |
| **PLANNED** | Targeted for the Capstone (prototype, deployment, demo) |
| **FUTURE** | Ideas for later. Not implemented, not promised |

> The project is in the **Capstone Scope Finalization** phase. The working prototype is the next milestone. Deployment is **planned, not completed**.

---

## 1. Overview

This project uses a **Convolutional Neural Network (CNN)** trained on the **PlantVillage** dataset to classify plant leaf images into **38 plant health and disease classes**.

It goes beyond a basic classifier. A normal classifier always returns one of its known classes, even when the image is a car, a laptop, a person, a blurry photo, a very dark photo, or an unsupported plant or disease. A high softmax probability does **not** mean the prediction is correct. It only means one of the 38 known classes received the largest share of probability.

This project adds a validation and decision layer around the CNN, built on one idea:

> **"Do not force the model to make a prediction when the available evidence is not reliable enough."**

The central question the system asks of every image:

> **"Does the system have enough evidence to provide a plant disease prediction for this particular image?"**

---

## 2. Problem Statement

A CNN trained on PlantVillage expects inputs from a similar visual domain. Real users upload arbitrary images, and a standard classifier may still assign one of its 38 classes to an unrelated or unreliable image. That is a **reliability problem**.

| Term | Meaning here |
|---|---|
| **Classification** | Assigning an image to one of the known classes |
| **Confidence** | The softmax probability of the top class. It is not the real-world chance of being correct |
| **Uncertainty** | How spread out or ambiguous the model output is |
| **Supported domain** | Leaf images of the 38 supported plant/condition classes |
| **Unsupported input** | Non-plant images, other species or diseases, poor-quality images, or images far from the training data |

The additional checks are **heuristic signals**. They **attempt to identify** unreliable inputs and **help reduce** misleading predictions. When evidence is weak, the **prediction may be withheld**.

> This project does **not** claim to solve out-of-distribution (OOD) detection.

---

## 3. Objectives

1. Classify plant leaf images into 38 supported classes using PlantVillage.
2. Build, train, and evaluate a CNN with a train/validation/test split.
3. Evaluate with accuracy, precision, recall, F1, confusion matrix, and per-class analysis.
4. Build a browser-based app for desktop and mobile with image upload.
5. Validate files and analyze image quality (size, blur, brightness, contrast).
6. Screen for plant-like content using a lightweight color heuristic.
7. Analyze confidence, prediction margin, entropy, and prediction consistency.
8. Withhold predictions with a clear reason when evidence is insufficient.
9. Show top-3 ranked outputs and supporting evidence for accepted predictions.
10. Deliver a working prototype, deployment, final documentation, and a demonstration *(PLANNED)*.

---

## 4. How the System Works

### Basic classifier vs. this system

```
BASIC CLASSIFIER
Image → CNN → Highest Probability → Display Result

THIS SYSTEM
Image → Validate → Check Quality → Check Plant-Likelihood → CNN
      → Confidence → Margin → Entropy → Consistency → Domain Check
      → Final Decision → ACCEPT or WITHHOLD
```

| Aspect | Basic classifier | This system |
|---|---|---|
| Non-plant image | May still output one of 38 classes | Attempts to identify and withhold |
| Blurry / dark / bright image | Outputs a label anyway | Checks quality, may withhold |
| Ambiguous output | Hidden | Analyzed with margin and entropy |
| Stability | Not checked | Checked with consistency test |
| Explanation | Usually none | Evidence or reason shown |
| Outcomes | One (a label) | Accept or Withhold |

### Pipeline

```
            User Image
                │
        File Validation
                │
      Image Quality Analysis
                │
   Plant / Non-Plant Screening
                │
        Preprocessing
                │
          CNN Inference
                │
 Confidence → Margin → Entropy
                │
      Prediction Consistency
                │
     Domain / OOD-related Checks
                │
         Final Decision
        ┌───────┴────────┐
        ▼                ▼
 Prediction Accepted   Prediction Withheld
 Class + Evidence      Reason + Guidance
```

> This is an academic project. It is **not** agriculturally certified and **not** claimed to be production-grade.

---

## 5. Dataset and Supported Classes

The project uses the **PlantVillage** dataset: labeled leaf images of healthy and diseased plants. It is used for training, validation, testing, and per-class analysis.

> **The full dataset is not committed to this repository** because of its size. No dataset URL is listed here because none was provided.

### 38 supported classes

| Plant | Supported conditions | Count |
|---|---|:---:|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy | 4 |
| Blueberry | Healthy | 1 |
| Cherry | Powdery Mildew, Healthy | 2 |
| Corn | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | 4 |
| Grape | Black Rot, Esca / Black Measles, Leaf Blight, Healthy | 4 |
| Orange | Haunglongbing / Citrus Greening | 1 |
| Peach | Bacterial Spot, Healthy | 2 |
| Pepper Bell | Bacterial Spot, Healthy | 2 |
| Potato | Early Blight, Late Blight, Healthy | 3 |
| Raspberry | Healthy | 1 |
| Soybean | Healthy | 1 |
| Squash | Powdery Mildew | 1 |
| Strawberry | Leaf Scorch, Healthy | 2 |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy | 10 |
| | **Total** | **38** |

Anything outside this list is **outside the supported domain**.

---

## 6. CNN Model and Training

### Model (conceptual level)

A CNN learns small filters that detect local visual patterns such as edges, textures, and lesion-like shapes. That makes it a good fit for leaf images.

| Component | Role |
|---|---|
| Convolution | Extracts local visual features |
| Pooling | Reduces spatial size and adds tolerance to small shifts |
| Dropout | Reduces overfitting during training |
| Dense layers | Combine features for classification |
| Softmax (38 outputs) | Turns scores into a probability for each class |

> Exact layer counts, filter sizes, and parameter counts are not stated here. Add them from the training code or model summary.

### Preprocessing

```
Open → Validate → Convert to RGB → Resize to 128×128 → float32 → Normalize → CNN
```

Inference preprocessing must stay **consistent** with the representation the model was trained on (size, channel order, data type, normalization).

### Data split

| Split | Share | Role |
|---|:---:|---|
| Training | 70% | Learn model weights |
| Validation | 15% | Monitor training, early stopping, checkpoint selection |
| Test | 15% | Final evaluation on held-out data |

Test data must **not** be used for training or model selection. Otherwise the test score is inflated.

### Training techniques

| Technique | Purpose |
|---|---|
| Random horizontal flip | Orientation tolerance |
| Random rotation | Tolerance to leaf angle |
| Random zoom | Tolerance to leaf size in frame |
| Early stopping (restore best weights) | Stop when validation stops improving |
| Model checkpointing | Save the best-performing model |

No other augmentation techniques are claimed.

---

## 7. Model Evaluation

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of test images classified correctly |
| Precision | Of images predicted as a class, how many truly belong to it |
| Recall | Of images truly in a class, how many were found |
| F1-score | Balance of precision and recall |
| Confusion matrix | Which classes get mixed up |
| Classification report | Per-class precision, recall, F1, and support |

**Recorded result:** the Week 6 model reached approximately **92.89% test accuracy** on the held-out PlantVillage test setup used during development.

> **This is not a real-world accuracy guarantee.** Real photos differ in lighting, background, camera quality, occlusion, multiple leaves, plant varieties, disease stages, and unseen domains.

### Why per-class evaluation matters

Overall accuracy can hide weak classes. Per-class precision, recall, F1, and support show which classes are strong, which are weak, and which get confused. Per-class numbers should be taken from the project's classification report and confusion matrix.

---

## 8. Reliability and Decision Layer

This layer is the core of the project. Every signal below is **one input** to the final decision, not a verdict on its own.

| Stage | What it checks | Why it matters |
|---|---|---|
| **File validation** | File exists, `.jpg` / `.jpeg` / `.png`, decodes as an image, size and dimensions | Stops broken or unsupported files early |
| **Image quality** | Width, height, sharpness, brightness, contrast | Poor images give unreliable predictions |
| **Blur detection** | Variance of the Laplacian against a configurable threshold | Sharp images have stronger edges, so higher variance |
| **Plant screening** | RGB-based vegetation ratio (green, yellow, brown regions) | Screens out obvious non-plant images |
| **Confidence** | Top-1 softmax probability | Weak top class suggests weak evidence |
| **Margin** | Gap between top-1 and top-2 | Small gap means ambiguity (e.g. 61% vs 58%, versus 88% vs 7%) |
| **Entropy** | Normalized entropy of the probability distribution | Spread-out distribution means higher uncertainty |
| **Consistency** | Prediction on original, horizontal flip, and two brightness variants | Stable predictions are stronger evidence |
| **Domain / OOD signals** | Combines all signals above | Heuristic estimate of whether the image fits the supported domain |

**Caveats**

- Laplacian variance and all thresholds are **heuristics**, not universal standards.
- The plant screen is **not** an object detector or segmentation model. It can be fooled by similar-colored objects.
- Consistency does **not** give robustness guarantees. A consistent prediction can still be wrong.
- This is **not** a perfect or formally guaranteed OOD detector.

### Decision outcomes

| Outcome | Prediction shown | Meaning | What the user should do |
|---|:---:|---|---|
| **Accepted** | Yes | Signals together support the prediction | Review result and evidence. Consult an expert for real decisions |
| **Poor Quality** | No | Too blurry, dark, bright, flat, or small | Upload a sharper, well-lit image |
| **Non-Plant** | No | Not enough plant-like content | Upload a clear leaf photo |
| **Uncertain** | No | Low confidence, small margin, high entropy, or inconsistent output | Try a closer image of a single leaf, plain background |
| **Out of Domain** | No | Appears outside supported plants or conditions | Check the supported class list |

The mapping from signals to outcomes uses configurable thresholds and may be refined during the Capstone.

### Evidence and Top-3

- **Evidence** for accepted results: image dimensions, sharpness, brightness, vegetation ratio, confidence, margin, entropy, consistency. This makes decisions inspectable.
- **Top-3** are ranked model outputs, **not** three confirmed diagnoses.

> Example format (illustrative only, not a measured result):
> `Tomato — Early Blight: 82.4%` · `Tomato — Late Blight: 8.7%` · `Tomato — Leaf Mold: 4.1%`

---

## 9. Web Application

### Technologies

| Layer | Technology |
|---|---|
| Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Model | TensorFlow / Keras |
| Image processing | OpenCV, Pillow, NumPy |

### Features

Image upload, drag and drop, preview, analyze button, loading indicator, result with confidence, top-3, evidence panel, withheld message with reason, try again, supported class list, and limitations info. The layout is designed to work on desktop and mobile browsers.

### Architecture

```
Browser (upload, preview, results)
        │  HTTP request with image
        ▼
Flask Backend
   File Validation → Image Quality → Plant Detector → Preprocessing
   → CNN → Prediction Analysis → Decision Layer
        │
        ▼  JSON response: Accepted / Withheld
Browser (shows result, evidence, or explanation)
```

The **browser** handles interaction and presentation. The **server** handles validation, model inference, and decisions.

### Backend components

| File | Responsibility |
|---|---|
| `app.py` | Flask app, routes, pipeline coordination |
| `config.py` | Central settings: paths, image size, class names, thresholds |
| `image_validator.py` | File-level validation |
| `quality_checker.py` | Dimensions, sharpness, brightness, contrast |
| `plant_detector.py` | Vegetation / color-based plant screening |
| `model_loader.py` | Loads the trained CNN |
| `prediction_service.py` | Preprocessing, inference, top-K, confidence, margin, entropy, consistency |
| `ood_detector.py` | Combines heuristic domain / uncertainty signals |

Centralized configuration (`config.py`) covers model path, image size, class names, allowed extensions, upload size, and the quality, plant, confidence, margin, entropy, consistency, and top-K settings. This makes thresholds easy to tune in one place. Numeric values are not listed here.

---

## 10. Project Structure

```
project/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── image_validator.py
├── model_loader.py
├── ood_detector.py
├── plant_detector.py
├── prediction_service.py
├── quality_checker.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── tests/
└── model/
```

> Representative structure. It may evolve during Capstone development. The dataset is not included.

---

## 11. Capstone Scope, Prototype and Deployment

This repository is the **Capstone project**. It does not restart the AI/ML journey. It **productizes and improves** existing work.

> **Capstone direction:** *"Develop a reliable AI-assisted plant disease image analysis system that combines CNN-based classification with image validation and uncertainty-aware prediction decisions through a user-friendly web interface."*

| Activity | Status |
|---|---|
| Finalize scope | In progress |
| Integrate existing components | Planned |
| Build working prototype | Planned (next milestone) |
| Improve UI/UX | Planned |
| Test complete system | Planned |
| Deploy | Planned (Render) |
| Final documentation | In progress |
| Final demonstration | Planned |

### Prototype flow (PLANNED)

```
Open app → Upload → Preview → Analyze → Validation → CNN → Decision Engine → Result
```

### Deployment (PLANNED)

The Capstone deployment phase will **target Render**. It has **not** been completed.

```
GitHub → Render → Build Environment → Dependencies → Flask App → Public URL
```

Deployment work includes production configuration, dependency management via `requirements.txt`, build and start commands, model file path, deployment testing, error checking, and handling memory and startup limits when loading a TensorFlow/Keras model.

---

## 12. Demonstration and Testing Plan

| Scenario | Input | Expected behavior |
|:---:|---|---|
| 1 | Valid plant image | **Accepted**: class, confidence, top-3, evidence |
| 2 | Non-plant image | **Withheld** (non-plant) |
| 3 | Poor-quality image | **Withheld** (poor quality) |
| 4 | Uncertain image | **Withheld** (uncertain) |

A normal classifier would give a disease label in all four cases. These scenarios show the system deciding whether it has enough evidence first.

### Testing categories

| Category | Expected behavior |
|---|---|
| Valid plant images | Accepted with evidence |
| Poor-quality images | Withheld as Poor Quality |
| Non-plant images | Withheld as Non-Plant, or flagged by other signals |
| Ambiguous images | Withheld as Uncertain |
| Unsupported / out-of-domain images | Withheld as Out of Domain where heuristics detect it |

> No test accuracy is claimed for the decision layer. Because it is heuristic, a bad image may be accepted or a good image withheld.

---

## 13. Limitations and Responsible Use

### Limitations

- The model is trained on **PlantVillage**, which may differ from real-world photos.
- Only **38 classes** are supported. Other plants and diseases are outside the domain.
- Real-world accuracy may be lower than the PlantVillage test result.
- The plant detector, quality thresholds, and OOD signals are all **heuristic**.
- Softmax confidence is not the real probability of being correct.
- No disease severity estimation is implemented.
- Outputs are **not** professional agricultural diagnosis.

### Responsible use

This is an **academic AI/ML Capstone project**. It should **not** replace agricultural experts, professional diagnosis, or field inspection.

The system deliberately favors:

> **"No prediction over an unjustified prediction."**

A confident but wrong label can mislead a user. Withholding is an intentional design choice that trades some convenience for caution and transparency.

---

## 14. Skills Used

These are skills developed and used in this project, not professional certifications.

| Area | Skills |
|---|---|
| Programming | Python, JavaScript, HTML, CSS |
| Data Science | Pandas, NumPy, data cleaning, data analysis |
| Machine Learning | Supervised learning, Linear Regression, classification, model evaluation, train/test split |
| NLP | Text preprocessing, TF-IDF, Naive Bayes |
| Deep Learning | TensorFlow, Keras, CNN, softmax, augmentation, early stopping, checkpointing |
| Computer Vision | Image preprocessing, RGB, resizing, blur, brightness and contrast analysis, vegetation detection |
| Model Evaluation | Accuracy, precision, recall, F1, confusion matrix, per-class evaluation, confidence, margin, entropy, consistency |
| Backend | Flask, file upload, request handling, JSON responses, validation, error handling |
| Frontend | Responsive HTML/CSS, JavaScript, drag and drop, image preview, API communication, dynamic results |
| Software Engineering | Modular Python, configuration, requirements management, testing, Git, GitHub, documentation |
| Deployment | Web deployment concepts, Render, production configuration, dependency management |

---

## 15. Future Scope

> **FUTURE only. None of these are implemented.**

| Area | Ideas |
|---|---|
| Data | Larger real-world dataset, more plant species and diseases |
| OOD / uncertainty | Embedding-based distance, deep ensembles, Monte Carlo dropout, energy-based methods |
| Explainability | Grad-CAM, attention visualization, saliency maps |
| Vision | Image segmentation, disease severity estimation |
| Platforms | Android / iOS apps or PWA |
| Content | Educational agricultural information |
| Maintenance | Continuous model improvement |

---

## 16. Roadmap and Current Status

### Project phases

| Phase | Name | Meaning | Status |
|:---:|---|---|---|
| 1 | Foundation & Model Development | Technical foundation and CNN development on PlantVillage | Completed (background) |
| 2 | Model Evaluation | Metrics, confusion matrix, per-class analysis | Completed (background) |
| 3 | Reliability & Validation Layer | Validation, quality, plant screening, uncertainty signals | Project basis, integration in Capstone |
| 4 | Web Application | Flask interface for upload, analysis, results | Project basis, refinement in Capstone |
| 5 | Capstone Prototype | One integrated, tested prototype with polished UI | **Planned (next)** |
| 6 | Deployment | Public URL, targeting Render | **Planned** |
| 7 | Final Polish & Demonstration | Final testing, documentation, demo | **Planned** |

### Long-term vision

```
Basic CNN Classifier → Validated CNN Classifier → Uncertainty-Aware Prediction
→ User-Friendly Application → Public Deployment → Explainable AI
→ Real-World Dataset Expansion → Advanced Plant Disease Analysis
```

The first three steps are the current basis. Deployment is planned. Explainable AI and dataset expansion are future scope.

### Current status

| Item | Status |
|---|---|
| Current phase | **Capstone Scope Finalization** |
| Current objective | Finalize the Capstone direction and system scope based on the completed AI/ML work |
| Next milestone | **Working Prototype** |
| Deployment | **Render** (planned, not completed) |
| Final deliverable | A documented, working, deployable plant disease detection and analysis web application with a final demonstration |

---

## 17. Conclusion

The project grew from data preparation, machine learning, NLP, deep learning, and computer vision into CNN training, evaluation, and per-class analysis. It then recognized that **accuracy on a held-out dataset is not the same as reliability on arbitrary user images**, and added image validation and uncertainty analysis. The Capstone brings these pieces into a user-facing web application, with deployment planned.

The goal is not merely to predict a disease. The goal is:

> **"To determine whether the system has enough evidence to provide a plant disease prediction for a given image."**

A normal classifier always tries to choose a class. This project can decide **not** to show a prediction when the evidence is insufficient.

---

## 18. Disclaimer

This project is an **academic AI/ML Capstone Project**. Predictions come from a machine learning model trained on a fixed dataset and are **not** professional agricultural diagnosis or expert advice.

The system may be wrong, especially for unsupported plant species, unsupported diseases, poor-quality images, out-of-domain images, and real-world images that differ significantly from the training data. For real agricultural decisions, consult qualified professionals.

---

<p align="center"><b>AI-Powered Plant Disease Detection & Analysis System</b><br>An academic AI/ML Capstone Project</p>

# LeafGuard AI — Plant Disease Detection & Analysis System

> An AI-powered plant leaf analysis application built using Convolutional Neural Networks (CNN), TensorFlow/Keras and Flask. The system combines image validation, quality analysis, plant-content screening and uncertainty-aware prediction decisions to help determine whether an uploaded leaf image provides sufficient evidence for a classification.

![Project Phase](https://img.shields.io/badge/Phase-Working%20Prototype-2E8B57)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-2E8B57)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JavaScript-orange)
![Deployment](https://img.shields.io/badge/Deployment-Render%20Planned-lightgrey)

---

## Table of Contents

1. Project Overview
2. Problem Statement
3. Project Objectives
4. Key Features
5. System Workflow
6. Dataset and Supported Classes
7. CNN Model and Training
8. Model Evaluation
9. Image Validation and Reliability Layer
10. Technology Stack
11. Application Architecture
12. Project Structure
13. Installation and Setup
14. Running the Application
15. Testing and Validation
16. Deployment
17. Limitations and Responsible Use
18. Skills and Concepts Applied
19. Future Enhancements
20. Development Roadmap
21. Current Project Status
22. Conclusion
23. Disclaimer

---

## 1. Project Overview

**LeafGuard AI** is an AI-powered plant disease detection and analysis system designed to analyze plant leaf images using deep learning and computer vision techniques.

The project uses a Convolutional Neural Network (CNN) trained on the PlantVillage dataset to classify images into 38 supported plant health and disease categories.

Unlike a basic image classifier that always returns the class with the highest probability, LeafGuard AI is designed with an additional validation and decision-making layer.

This layer examines image quality, plant-like visual content, prediction confidence, prediction margin, entropy and prediction consistency.

The system aims to make one of two decisions:

- **Accept Prediction:** Display the predicted plant condition when the available evidence satisfies the configured decision criteria.
- **Withhold Prediction:** Avoid displaying an unsupported classification when the image or prediction does not meet those criteria.

The main principle behind the project is:

> A machine learning model should not be forced to provide a prediction when the available evidence is insufficient.

### Project Information

| Item | Description |
|---|---|
| Project Name | LeafGuard AI |
| Project Title | AI-Powered Plant Disease Detection & Analysis System |
| Project Domain | Artificial Intelligence, Machine Learning and Computer Vision |
| Main Approach | CNN-based Image Classification |
| Dataset | PlantVillage |
| Backend | Python and Flask |
| Frontend | HTML, CSS and JavaScript |
| Deep Learning Framework | TensorFlow / Keras |
| Current Development Stage | Working Local Web Prototype |
| Deployment Target | Render |
| Project Type | Academic AI/ML Project |

## 2. Problem Statement

Plant diseases can affect agricultural productivity and crop quality. Identifying disease symptoms from leaf images is one potential application of computer vision and deep learning.

However, a standard image classification model has an important limitation.

A CNN trained on a fixed dataset generally selects one of its known output classes even when the uploaded image is unrelated to its training data.

For example, a user might upload:

- A clear plant leaf image.
- A blurry or dark leaf image.
- A photograph of a person.
- An image of a vehicle.
- An unsupported plant species.
- A leaf with an unsupported disease.

A conventional classifier may still return one of its trained classes.

The highest softmax probability does not guarantee that the classification is correct.

Therefore, this project introduces additional image validation and prediction reliability checks before presenting a classification.

**The problem addressed by LeafGuard AI is not only how to classify an image, but also how to evaluate whether a classification should be presented at all.**

## 3. Project Objectives

The major objectives are:

1. Develop a CNN-based plant disease classification system.
2. Use the PlantVillage dataset for supervised image classification.
3. Train and evaluate a deep learning model across 38 supported classes.
4. Apply image preprocessing consistently during training and inference.
5. Develop a browser-based application for uploading plant leaf images.
6. Implement file validation and image-quality analysis.
7. Examine plant-like visual content using a lightweight screening approach.
8. Analyze prediction confidence and the difference between the top predictions.
9. Examine prediction uncertainty using entropy.
10. Evaluate prediction consistency across modified image inputs.
11. Introduce an uncertainty-aware decision layer.
12. Provide explanations when a prediction is withheld.
13. Integrate the trained model with a Flask backend.
14. Test the complete image-upload and prediction workflow.
15. Prepare the application for public deployment.

## 4. Key Features

### 4.1 Web-Based Image Analysis

- Browser-based application.
- Plant leaf image upload.
- Image preview.
- User-friendly analysis interface.
- Responsive visual layout.

### 4.2 CNN-Based Classification

- Deep learning model using TensorFlow/Keras.
- PlantVillage-based classification.
- 38 supported output categories.
- Prediction confidence analysis.

### 4.3 Image Validation

- File-format validation.
- Image decoding checks.
- Image dimensions and size checks.
- Image-quality analysis.

### 4.4 Reliability and Uncertainty Analysis

The project includes separate components for:

- Image-quality checking.
- Plant-content screening.
- Model loading.
- Prediction processing.
- Uncertainty and domain-related decision analysis.

### 4.5 Prediction Decision

The system is designed around two possible outcomes:

| Outcome | Description |
|---|---|
| Accepted | A prediction is displayed when configured decision criteria are satisfied. |
| Withheld | A prediction is not presented when the available evidence is insufficient. |

The exact behavior and reliability of each decision must be established through application testing.

## 5. System Workflow

The overall system follows this pipeline:

```text
                USER
                  |
                  v
          Upload Leaf Image
                  |
                  v
            File Validation
                  |
                  v
        Image Quality Analysis
                  |
                  v
        Plant Content Screening
                  |
                  v
          Image Preprocessing
                  |
                  v
          CNN Model Inference
                  |
                  v
       Prediction Probability Analysis
                  |
          +-------+-------+
          |       |       |
          v       v       v
      Confidence Margin Entropy
                  |
                  v
        Prediction Consistency
                  |
                  v
        Reliability / Domain Checks
                  |
                  v
          Final Decision Layer
             /          \
            /            \
           v              v
    ACCEPT PREDICTION   WITHHOLD
           |              |
           v              v
    Predicted Class    Explanation
    Confidence         and Guidance
    Supporting Data
```

The pipeline combines deep learning classification with additional checks intended to reduce misleading predictions.

These checks are heuristic signals and do not provide a formal guarantee that an image is inside or outside the training distribution.

## 6. Dataset and Supported Classes

### 6.1 PlantVillage Dataset

The project uses the PlantVillage dataset, which contains labeled images of healthy and diseased plant leaves.

The dataset provides the image examples required for supervised CNN training and evaluation.

The full dataset is not included in this GitHub repository because of its size.

### 6.2 Supported Classes

The model covers 38 plant health and disease categories.

| Plant | Supported Conditions | Number of Classes |
|---|---|---:|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy | 4 |
| Blueberry | Healthy | 1 |
| Cherry | Powdery Mildew, Healthy | 2 |
| Corn | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | 4 |
| Grape | Black Rot, Esca / Black Measles, Leaf Blight, Healthy | 4 |
| Orange | Huanglongbing / Citrus Greening | 1 |
| Peach | Bacterial Spot, Healthy | 2 |
| Pepper Bell | Bacterial Spot, Healthy | 2 |
| Potato | Early Blight, Late Blight, Healthy | 3 |
| Raspberry | Healthy | 1 |
| Soybean | Healthy | 1 |
| Squash | Powdery Mildew | 1 |
| Strawberry | Leaf Scorch, Healthy | 2 |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy | 10 |
| **Total** | **38 supported classes** | **38** |

Images of unsupported plants and diseases are outside the model's known classification categories.

### 6.3 Dataset Considerations

- Dataset images are used for model development and evaluation.
- The dataset is not automatically downloaded by the web application.
- The original dataset is excluded from GitHub because of its size.
- Real-world images may differ from PlantVillage images in lighting, background, camera quality and disease appearance.

## 7. CNN Model and Training

### 7.1 Convolutional Neural Network

A Convolutional Neural Network is a deep learning architecture commonly used for image recognition and classification.

It learns visual features from images through layers of mathematical operations.

The conceptual components include:

| Component | Purpose |
|---|---|
| Convolution Layers | Extract visual features such as edges and textures |
| Activation Functions | Introduce non-linearity |
| Pooling Layers | Reduce spatial dimensions |
| Dropout | Help reduce overfitting |
| Dense Layers | Combine extracted features |
| Softmax Output | Produce class probability scores |

The output layer represents the 38 supported classes.

### 7.2 Image Preprocessing

The inference pipeline is designed around the model's input representation.

The documented preprocessing format is:

```text
Input Image
    |
    v
Image Validation
    |
    v
Convert to RGB
    |
    v
Resize to 128 x 128
    |
    v
Convert to float32
    |
    v
Normalize Pixel Values
    |
    v
CNN Model
```

The preprocessing performed during inference must remain consistent with the preprocessing used during model training.

### 7.3 Dataset Splitting

The training methodology uses the following intended data split:

| Dataset Split | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Learn model parameters |
| Validation | 15% | Monitor training and select model checkpoints |
| Testing | 15% | Evaluate final performance |

The test dataset should remain separate from training and model-selection activities.

### 7.4 Training Techniques

The earlier model-development work includes the following techniques:

- Random horizontal flipping.
- Random rotation.
- Random zoom.
- Early stopping.
- Model checkpointing.

These techniques help improve the model's ability to learn useful visual patterns and reduce overfitting.

## 8. Model Evaluation

The CNN model was evaluated using standard classification metrics.

### 8.1 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Percentage of correctly classified test images |
| Precision | Correct positive predictions relative to all positive predictions |
| Recall | Correctly identified positive samples relative to actual positive samples |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect class predictions |
| Classification Report | Provides per-class precision, recall, F1-score and support |

### 8.2 Previously Recorded Model Result

The earlier Week 6 model-development work recorded approximately:

**92.89% test accuracy**

This result belongs to the held-out PlantVillage evaluation setup used during that development stage.

It should not be interpreted as guaranteed accuracy on arbitrary real-world plant photographs.

The current Week 8 web application must be tested independently to verify that it loads the intended model and produces predictions correctly.

### 8.3 Importance of Per-Class Evaluation

Overall accuracy does not explain how well the model performs on every category.

Per-class precision, recall and F1-score help identify:

- Strongly classified categories.
- Weakly classified categories.
- Frequently confused plant conditions.
- Categories that require further investigation.

The reliability of the complete application depends on both classification performance and the additional decision layer.

## 9. Image Validation and Reliability Layer

This is an important part of the LeafGuard AI design.

Instead of relying only on the CNN's highest probability, the system includes additional analysis components.

### 9.1 File Validation

The image validator is responsible for checking uploaded files.

Potential checks include:

- Whether a file was provided.
- Whether the extension is supported.
- Whether the image can be decoded.
- Whether the image dimensions are acceptable.
- Whether the file size is within configured limits.

Supported formats are intended to include JPG, JPEG and PNG.

### 9.2 Image Quality Analysis

Image quality can affect model predictions.

The quality checker considers image characteristics such as:

| Signal | Purpose |
|---|---|
| Image Dimensions | Checks whether the image has sufficient resolution |
| Blur | Examines sharpness |
| Brightness | Detects very dark or bright images |
| Contrast | Examines differences between image regions |

Variance of the Laplacian can be used as a heuristic for blur detection.

However, the appropriate thresholds depend on the image domain and must be validated.

### 9.3 Plant Content Screening

The plant detector uses lightweight visual or color-based analysis to estimate whether an image contains plant-like content.

For example, RGB-based color analysis may examine green, yellow and brown regions.

This is a heuristic screening technique, not a trained object detector or a plant segmentation model.

### 9.4 Prediction Confidence

The model produces a probability distribution over the supported classes.

The highest probability represents the top predicted class.

However:

> A softmax score is not a calibrated probability that the prediction is correct.

Confidence must therefore be considered alongside other evidence.

### 9.5 Prediction Margin

Prediction margin is the difference between the highest and second-highest prediction scores.

For example:

```text
Example A:
Class 1 = 88%
Class 2 = 7%
Margin = 81 percentage points

Example B:
Class 1 = 61%
Class 2 = 58%
Margin = 3 percentage points
```

A smaller margin may indicate ambiguity between classes.

### 9.6 Entropy

Entropy measures the spread of the model's probability distribution.

A more distributed probability output generally indicates greater uncertainty.

Entropy is used as another signal in the decision process.

### 9.7 Prediction Consistency

The system design includes checking predictions across modified versions of an image, such as:

- Original image.
- Horizontally flipped image.
- Brightness-adjusted image variants.

The objective is to examine whether the predicted class remains consistent under these changes.

Consistency is not a guarantee of correctness.

### 9.8 Domain and Uncertainty Analysis

The project includes an OOD-related analysis component that combines signals from image validation, image quality and model prediction analysis.

It should be treated as a heuristic decision layer rather than a formally validated out-of-distribution detection system.

### 9.9 Possible Decision Outcomes

| Outcome | Meaning |
|---|---|
| Accepted | Prediction satisfies the configured decision criteria |
| Poor Quality | Image quality is insufficient |
| Non-Plant | Plant-like content screening does not satisfy its criteria |
| Uncertain | Prediction evidence is insufficient |
| Out of Domain | Heuristic checks indicate a potentially unsupported input |

The final decision depends on the configured thresholds and implementation.

## 10. Technology Stack

| Category | Technology | Usage |
|---|---|---|
| Programming Language | Python | Backend and ML development |
| Frontend | HTML5 | Application structure |
| Styling | CSS3 | Responsive user interface |
| Client-Side Logic | JavaScript | Image upload and interaction |
| Backend Framework | Flask | HTTP routes and application logic |
| Deep Learning | TensorFlow | Neural network operations |
| Model API | Keras | Model loading and inference |
| Image Processing | Pillow | Image reading and processing |
| Numerical Processing | NumPy | Numerical operations |
| Machine Learning | Scikit-learn | Evaluation and supporting ML tasks in the wider project |
| Version Control | Git | Source control |
| Repository Hosting | GitHub | Project hosting |
| Deployment Target | Render | Planned public web hosting |

## 11. Application Architecture

### 11.1 Frontend

The frontend provides the user interface for interacting with the application.

Its responsibilities include:

- Displaying the LeafGuard AI homepage.
- Allowing image selection.
- Previewing the selected image.
- Communicating with the backend.
- Displaying available analysis results.

### 11.2 Flask Backend

The Flask backend handles the application logic and coordinates the image-analysis pipeline.

Its responsibilities include:

- Handling HTTP requests.
- Receiving uploaded images.
- Validating uploaded files.
- Calling the relevant image-analysis components.
- Loading and using the trained model.
- Returning analysis results to the frontend.

### 11.3 Machine Learning Integration

The trained CNN model is loaded by the backend through the model-loading component.

The prediction service handles preprocessing and model inference.

The additional validation and decision components are used to examine the reliability of the input and prediction.

### 11.4 Architecture Diagram

```text
+------------------------------------+
|             FRONTEND               |
|                                    |
|       HTML + CSS + JavaScript      |
|                                    |
|       Image Upload and Preview     |
+------------------+-----------------+
                   |
                   | HTTP Request
                   v
+------------------------------------+
|            FLASK BACKEND            |
|                                    |
|             app.py                 |
|                                    |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
|        IMAGE VALIDATION            |
|                                    |
|        image_validator.py          |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
|       IMAGE QUALITY ANALYSIS       |
|                                    |
|        quality_checker.py          |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
|        PLANT CONTENT CHECK         |
|                                    |
|         plant_detector.py          |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
|         MODEL INFERENCE            |
|                                    |
|          model_loader.py           |
|        prediction_service.py       |
|                                    |
|       TensorFlow / Keras CNN        |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
|      RELIABILITY ANALYSIS          |
|                                    |
|           ood_detector.py          |
|                                    |
+------------------+-----------------+
                   |
                   v
+------------------------------------+
|          FINAL RESPONSE            |
|                                    |
|      Prediction / Withholding      |
+------------------------------------+
```

## 12. Project Structure

The repository contains the earlier AI/ML learning work and the Week 8 web application.

The relevant Week 8 structure is:

```text
AI-ML-Foundation/
|
|-- week6/
|   |-- models/
|       |-- plant_disease_cnn_week6.keras
|
|-- week7/
|
|-- week8/
    |
    |-- app.py
    |-- config.py
    |-- image_validator.py
    |-- model_loader.py
    |-- ood_detector.py
    |-- plant_detector.py
    |-- prediction_service.py
    |-- quality_checker.py
    |
    |-- requirements.txt
    |-- pytest.ini
    |-- README.md
    |-- .gitignore
    |
    |-- static/
    |   |-- style.css
    |   |-- script.js
    |
    |-- templates/
    |   |-- index.html
    |
    |-- tests/
    |
    |-- .venv/
    |-- __pycache__/
```

**Note:** This is a representative structure based on the current project layout. The exact contents of the static, templates and tests directories may change as development continues.

The `.venv` and `__pycache__` directories are local development artifacts and should not be committed to GitHub.

The Week 8 application also relies on the trained model from the earlier Week 6 work. Its model path must be configured correctly for local execution and deployment.

## 13. Installation and Setup

Follow these instructions to run the project locally.

### 13.1 Prerequisites

Install the following:

- Python 3.
- Git.
- Visual Studio Code (recommended).
- A web browser.
- GitHub repository access.

The current development environment uses Python 3.13.14.

### 13.2 Clone the Repository

```bash
git clone https://github.com/mohd-ibrahim-prog/AI-ML-Foundation.git
```

Navigate into the repository:

```bash
cd AI-ML-Foundation
```

Enter the Week 8 directory:

```bash
cd week8
```

### 13.3 Create a Virtual Environment

```bash
python -m venv .venv
```

### 13.4 Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 13.5 Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 13.6 Verify the Installation

```bash
python --version
```

Check TensorFlow:

```bash
python -m pip show tensorflow-cpu
```

The actual dependency versions should be taken from the current `requirements.txt` file.

## 14. Running the Application

After completing installation, start the Flask application:

```bash
python app.py
```

If the application starts successfully, Flask displays its local development address.

Open:

```text
http://127.0.0.1:5000/
```

The application should display the LeafGuard AI homepage.

### Current Local Execution Evidence

The Week 8 application has successfully:

- Started using Flask.
- Served the homepage with HTTP 200.
- Served `static/script.js` with HTTP 200.
- Served `static/style.css` with HTTP 200.
- Displayed the LeafGuard AI user interface in a browser.

These checks establish that the local homepage and its static assets load successfully.

They do not independently verify the complete CNN prediction workflow.

### Stopping the Application

To stop the local Flask server, press:

```text
CTRL + C
```

## 15. Testing and Validation

Testing is essential to verify that the complete application works correctly.

### 15.1 Application Testing

| Test | Purpose |
|---|---|
| Homepage Loading | Verify that the main page opens |
| Static Assets | Verify CSS and JavaScript loading |
| Image Upload | Verify image submission |
| Image Preview | Verify frontend preview |
| Backend Processing | Verify uploaded image handling |
| Model Loading | Verify trained model loading |
| Prediction | Verify CNN inference |
| Error Handling | Verify invalid input handling |

### 15.2 Image Testing Scenarios

| Scenario | Input | Expected Behavior |
|---|---|---|
| Valid Leaf Image | Supported plant leaf image | Analyze and return a prediction when criteria are satisfied |
| Non-Plant Image | Person, vehicle or unrelated object | Withhold or flag the prediction when screening detects the unsupported input |
| Blurry Image | Out-of-focus leaf image | Apply image-quality checks |
| Dark Image | Very low-brightness image | Apply brightness checks |
| Unsupported Format | PDF or other unsupported file | Reject the upload |
| Corrupted Image | Invalid image data | Handle decoding failure |
| Ambiguous Prediction | Similar scores for multiple classes | Apply uncertainty analysis |
| Unsupported Plant | Plant outside the supported classes | Apply domain-related checks |

### 15.3 Testing Framework

Pytest is included in the project structure.

Run the test suite from the Week 8 directory:

```bash
python -m pytest -v
```

A successful test report should be recorded with the actual number of tests passed, failed or skipped.

**Testing status:** The homepage and static-file responses have been observed successfully. Full prediction testing and the complete automated test results still need to be verified.

## 16. Deployment

The deployment target for LeafGuard AI is **Render**.

The application is currently running locally. Public deployment is the next development milestone.

### 16.1 Why Render?

Render is suitable for deploying Python web applications and Flask backends.

It supports connecting a GitHub repository, installing Python dependencies and running a web service.

### 16.2 Planned Deployment Architecture

```text
GitHub Repository
       |
       v
Render Web Service
       |
       v
Install Python Dependencies
       |
       v
Load Flask Application
       |
       v
Load CNN Model
       |
       v
Public Web Application
```

### 16.3 Planned Render Configuration

| Setting | Value |
|---|---|
| Platform | Render |
| Service Type | Web Service |
| Repository | AI-ML-Foundation |
| Root Directory | week8 |
| Runtime | Python |
| Build Command | pip install -r requirements.txt |
| Start Command | gunicorn app:app --bind 0.0.0.0:$PORT |

The start command assumes the Flask application object is named `app` inside `app.py`. This must be verified against the actual source code.

### 16.4 Deployment Considerations

Before deployment, the following must be checked:

- Correct model path.
- Availability of the trained model in the deployment environment.
- Correct Python and package compatibility.
- Production WSGI server configuration.
- Upload-size restrictions.
- Memory usage during TensorFlow model loading.
- Startup time.
- Error handling.
- Successful prediction through the public URL.

The model file must be accessible to the deployed application. A local model path that works on Windows will not automatically work on Render.

**Deployment Status:** Planned. A public deployment URL will be added after successful deployment and testing.

## 17. Limitations and Responsible Use

### 17.1 Current Limitations

1. The model is trained on the PlantVillage dataset.
2. The model supports only 38 known categories.
3. Real-world photographs may differ from training images.
4. Softmax confidence is not a guarantee of correctness.
5. Color-based plant screening is heuristic.
6. Blur, brightness and contrast thresholds require validation.
7. Prediction consistency does not guarantee a correct result.
8. The OOD-related component is not a formally validated OOD detector.
9. Disease severity estimation is not part of the current documented scope.
10. The system is not a substitute for expert agricultural diagnosis.

### 17.2 Responsible Use

LeafGuard AI is an academic AI/ML project.

The output should be interpreted as an AI-assisted classification result rather than a confirmed diagnosis.

For important agricultural decisions, users should consult qualified agricultural professionals.

The project's design philosophy is:

> It is better to withhold an unsupported prediction than to present an unjustified result as certain.

## 18. Skills and Concepts Applied

The project combines concepts from programming, machine learning, deep learning, computer vision and web development.

| Area | Skills and Concepts |
|---|---|
| Programming | Python, JavaScript |
| Data Processing | NumPy, Pandas |
| Machine Learning | Supervised learning, classification, data splitting |
| Deep Learning | CNN, TensorFlow, Keras, Softmax |
| Computer Vision | Image preprocessing, RGB, image resizing |
| Image Analysis | Blur, brightness, contrast and color analysis |
| Model Evaluation | Accuracy, precision, recall, F1-score, confusion matrix |
| Prediction Analysis | Confidence, margin, entropy and consistency |
| Backend Development | Flask, HTTP requests, file uploads |
| Frontend Development | HTML, CSS, JavaScript |
| Software Engineering | Modular code, configuration, requirements management |
| Testing | Pytest and application validation |
| Version Control | Git and GitHub |
| Deployment | Render and production application configuration |

## 19. Future Enhancements

The following are potential improvements and are not claimed as completed features.

| Area | Future Enhancement |
|---|---|
| Dataset | Add more real-world plant images |
| Classification | Expand supported plant species and disease categories |
| Model Improvement | Experiment with transfer learning and improved CNN architectures |
| Explainable AI | Integrate Grad-CAM visual explanations |
| Uncertainty | Explore calibrated confidence and more advanced uncertainty methods |
| OOD Detection | Evaluate embedding-based and other formal OOD detection approaches |
| Computer Vision | Add disease-region segmentation |
| Disease Analysis | Explore disease severity estimation |
| User Experience | Improve reporting and result presentation |
| Platform | Explore Progressive Web App or mobile application support |
| Deployment | Improve monitoring, performance and scalability |

## 20. Development Roadmap

The project builds on the AI/ML foundation developed during Weeks 1–7.

| Phase | Development Activity | Status |
|---|---|---|
| Week 1 | Python and data science fundamentals | Earlier work |
| Week 2 | Data analysis and preprocessing | Earlier work |
| Week 3 | Machine learning fundamentals | Earlier work |
| Week 4 | PlantVillage dataset preparation and initial deep learning work | Earlier work |
| Week 5 | CNN model development and training | Earlier work |
| Week 6 | Model improvement and evaluation | Earlier work |
| Week 7 | Model analysis and per-class evaluation | Earlier work |
| Week 8 | Flask web application and AI integration | Local application running; full workflow verification pending |
| Week 9 | Capstone integration and refinement | Planned |
| Week 10 | Prototype testing and improvements | Planned |
| Week 11 | Public deployment and deployment testing | Planned |
| Week 12 | Final documentation, polish and demonstration | Planned |

The weekly activities describe the broader learning and project-development journey. They do not imply that every planned milestone has already been completed.

## 21. Current Project Status

### Development Status

| Component | Current Status |
|---|---|
| AI/ML Foundation | Earlier work completed |
| PlantVillage Dataset Work | Earlier work completed |
| CNN Model Development | Earlier work completed |
| Week 6 Model Evaluation | Earlier results recorded |
| Per-Class Model Analysis | Earlier work completed |
| Week 8 Flask Backend | Application starts locally |
| LeafGuard AI Homepage | Loads locally |
| CSS and JavaScript | Static files return HTTP 200 |
| Model Integration | Implemented components present; end-to-end prediction needs verification |
| Image Upload Workflow | Full workflow needs verification |
| Prediction Reliability | Full behavior needs verification |
| Automated Testing | Results need verification |
| GitHub | Existing repository |
| Render Deployment | Planned |
| Final Demonstration | Planned |

### Current Milestone

**Week 8 — Local Web Application**

The Flask development server successfully starts and serves the LeafGuard AI homepage.

The next milestone is to verify actual plant-image predictions, model availability, upload validation and error handling.

After successful local testing, the application will be prepared for Render deployment.

### Repository

GitHub:  
https://github.com/mohd-ibrahim-prog/AI-ML-Foundation

## 22. Conclusion

LeafGuard AI brings together machine learning, deep learning, computer vision and web development to create an AI-assisted plant disease analysis application.

The project began with data preparation and machine learning fundamentals, progressed through CNN development and evaluation, and now includes a Flask-based web application.

Its central contribution is the attempt to go beyond conventional classification by considering image quality, prediction confidence and uncertainty before presenting a result.

The objective is not simply to make a model predict a disease.

It is to build a system that considers whether the available image provides sufficient evidence for a prediction.

> **LeafGuard AI — Smarter Plant Analysis Through AI, Validation and Responsible Predictions.**

## 23. Disclaimer

This is an academic AI/ML project developed for educational and demonstration purposes.

The model's predictions are based on its training data and supported classification categories. They may be inaccurate, especially for unsupported plant species, diseases, poor-quality images and real-world images that differ from the training dataset.

The application should not be treated as a professional agricultural diagnosis system.

Always consult qualified agricultural professionals before making important crop-treatment or agricultural decisions.

---

<p align="center">
  <strong>LeafGuard AI</strong><br>
  AI-Powered Plant Disease Detection & Analysis System<br>
  Academic AI/ML Project
</p>

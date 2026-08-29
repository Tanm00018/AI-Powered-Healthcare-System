# AI-Powered Healthcare System

> **Machine Learning–Driven Healthcare Platform for Symptom Analysis and Medical Record Management**

An interactive healthcare platform that combines **machine learning-based disease prediction** with patient and doctor workflows for managing medical information.

The system provides separate patient and doctor portals, allowing patients to maintain their medical history and perform AI-based symptom analysis, while doctors can review patient records, track treatment history, prescribe medications, record allergies, and attach medical reports.

---

## 1. Project Overview

Healthcare applications often involve two distinct challenges: making information accessible to patients and organizing clinical information for healthcare providers.

This project explores a lightweight approach to both problems by combining an interactive **Streamlit application** with a **Random Forest classification model** for symptom-based disease prediction.

The system brings together:

* Machine learning-based disease prediction
* Symptom feature processing
* Patient and doctor authentication
* Role-based application flows
* Medical history management
* Treatment and medication records
* Allergy tracking
* Medical report attachments
* Interactive patient summaries
* Doctor-side patient management

The result is a unified healthcare application where the machine-learning component is integrated directly into a broader patient-management workflow.

---

## 2. Key Capabilities

### 2.1 AI Symptom Analysis

The patient portal provides an interactive symptom-analysis interface.

Users enter symptoms as a comma-separated list:

```text
fever, headache, cough
```

The application normalizes the input and converts it into a binary feature vector corresponding to the symptoms recognized by the trained model.

```text
User Symptoms
      ↓
Input Normalization
      ↓
Symptom Feature Vector
      ↓
Random Forest Classifier
      ↓
Predicted Condition
```

The trained classifier then generates a predicted disease condition.

---

### 2.2 Machine Learning Disease Prediction

The prediction engine is based on a **Random Forest Classifier** implemented using Scikit-learn.

The training pipeline:

```text
Dataset
   ↓
Dataset Sampling
   ↓
Feature / Target Separation
   ↓
Train / Test Split
   ↓
Random Forest Training
   ↓
Model Evaluation
   ↓
Serialized Model
```

The training script uses:

```text
RandomForestClassifier
n_estimators = 100
random_state = 42
```

The model is trained on the project's augmented disease dataset and evaluated using a held-out test split.

---

### 2.3 Patient Portal

Patients have access to a dedicated portal containing:

* Medical summary
* Known allergies
* Current medications
* AI symptom analysis
* Medical history folders
* Doctor treatment records
* Uploaded medical reports

Medical records are organized into folders and displayed chronologically, with the newest records appearing first.

---

### 2.4 Doctor Portal

Doctors have a separate workflow for managing patient information.

The doctor portal provides:

* Patient selection
* Patient medical summaries
* Allergy information
* Treatment history
* Medical record folders
* Doctor-based record filtering
* Treatment-plan entry
* Medication recording
* Medical report uploads

This creates a role-specific interface rather than exposing the same workflow to every user.

---

### 2.5 Medical Record Management

Each medical record can contain:

```text
Record ID
Doctor
Timestamp
Allergies
Medications
Treatment Plan
Uploaded Report
```

Records can be grouped into custom medical-history folders, allowing information to be organized around different medical contexts.

Uploaded reports support:

```text
PDF
JPG
JPEG
PNG
```

The application can display image attachments directly and render PDF documents within the interface.

---

### 2.6 Authentication and Roles

The application implements separate **Patient** and **Doctor** roles.

During signup, users select their role and provide a password.

Passwords are hashed using **bcrypt** before being stored in the application's session state.

```text
Signup
   ↓
Password Validation
   ↓
bcrypt Hashing
   ↓
User Registration
   ↓
Login
   ↓
Role Detection
   ↓
Patient / Doctor Portal
```

The application also validates password length and prevents duplicate usernames.

---

## 3. System Architecture

```text
                         USER
                          │
              ┌───────────┴───────────┐
              │                       │
           Patient                  Doctor
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    Streamlit    │
                 │   Application   │
                 └────────┬────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
       Authentication   Patient      Doctor
                         Portal       Portal
            │             │             │
            │             ├─────────────┤
            │             │             │
            │             ▼             ▼
            │      Medical Records   Patient Data
            │             │
            │             ▼
            │      Medical Reports
            │
            ▼
      Role-based Access
                          │
                          ▼
                 ┌─────────────────┐
                 │  AI Symptom     │
                 │    Analysis     │
                 └────────┬────────┘
                          │
                          ▼
                 Symptom Feature Vector
                          │
                          ▼
                 ┌─────────────────┐
                 │  Random Forest  │
                 │    Classifier   │
                 └────────┬────────┘
                          │
                          ▼
                  Predicted Disease
```

---

## 4. Machine Learning Pipeline

The machine-learning component is implemented in `train_model.py`.

### 4.1 Dataset Preparation

The training script loads:

```text
data/Final_Augmented_Dataset.csv
```

The dataset is sampled to 10,000 records using a fixed random seed:

```python
df.sample(n=10000, random_state=42)
```

The target column is:

```text
diseases
```

All remaining columns are used as input features.

```text
X = Symptoms / Features
y = Disease
```

---

### 4.2 Train / Test Split

The dataset is divided into training and testing subsets using an 80/20 split.

```text
80% → Training
20% → Testing
```

A fixed `random_state=42` is used to make the split reproducible.

---

### 4.3 Random Forest Model

The classifier is implemented using Scikit-learn's `RandomForestClassifier`.

```text
Random Forest
├── 100 decision trees
├── random_state = 42
└── classification-based prediction
```

The model learns relationships between symptom features and disease labels from the training data.

---

### 4.4 Model Evaluation

After training, predictions are generated on the held-out test set.

The training pipeline reports:

* Classification report
* Confusion matrix

```text
Test Data
    ↓
Trained Model
    ↓
Predictions
    ↓
Classification Report
    +
Confusion Matrix
```

This provides visibility into classification performance across the disease classes.

---

### 4.5 Model Serialization

After training, the Random Forest model is serialized using Python's `pickle` module.

```text
model/disease_predictor.pkl
```

The application loads this trained model during startup and uses it for inference.

---

## 5. Prediction Pipeline

During a patient's AI assessment, the application follows this pipeline:

```text
User Input
    ↓
Comma-Separated Symptoms
    ↓
Lowercase + Whitespace Normalization
    ↓
Symptom Matching
    ↓
Binary Feature Vector
    ↓
Random Forest Model
    ↓
Disease Prediction
    ↓
Streamlit Result
```

Each recognized symptom is represented as a binary value:

```text
1 → Symptom present
0 → Symptom absent
```

The resulting vector is reshaped into the format expected by the trained classifier before inference.

---

## 6. Patient Data Workflow

The patient side of the application is centered around maintaining a consolidated view of medical information.

### Medical Summary

The application aggregates information across medical-history folders to surface:

```text
Known Allergies
Current Medications
```

Records are sorted by timestamp so that recent information is surfaced first.

### Medical History

Patients can navigate through their medical-history folders and inspect individual records.

Each record can include:

```text
Doctor
Date
Treatment
Medications
Allergies
Medical Report
```

This provides a structured representation of the patient's previous medical interactions.

---

## 7. Doctor Workflow

The doctor portal is designed around patient-specific record management.

```text
Doctor Login
     ↓
Patient Selection
     ↓
Patient Summary
     ↓
Medical History
     ↓
Treatment / Medication Entry
     ↓
Report Upload
     ↓
Updated Medical Record
```

Doctors can also filter records by doctor when reviewing a patient's medical history.

New records are assigned a unique identifier using UUID generation.

---

## 8. Application Architecture

The current application is implemented as a single Streamlit application with functional separation between the main workflows.

```text
app.py
│
├── Authentication
│   ├── signup()
│   ├── login()
│   └── logout_button()
│
├── Patient Workflow
│   └── patient_portal()
│
├── Doctor Workflow
│   └── doctor_portal()
│
├── Medical Records
│   └── display_record()
│
└── Application Entry
    └── main_app()
```

The ML training workflow is separated into:

```text
train_model.py
```

This keeps model training independent from application inference.

---

## 9. Project Structure

```text
AI-Powered-Healthcare-System/
│
├── app.py
├── train_model.py
├── data/
│   └── Final_Augmented_Dataset.csv
│
├── model/
│   └── disease_predictor.pkl
│
├── vectorizer.pkl
├── README.md
└── LICENSE
```

### `app.py`

Contains the Streamlit application, authentication workflow, patient portal, doctor portal, medical-record management, report handling, and model inference.

### `train_model.py`

Contains the machine-learning training pipeline, dataset preparation, Random Forest training, evaluation, and model serialization.

### `disease_predictor.pkl`

Serialized Random Forest model used by the application for disease prediction.

### `vectorizer.pkl`

Serialized feature representation used by the application to obtain the model's recognized symptom features.

---

## 10. Technology Stack

| Layer                | Technology    |
| -------------------- | ------------- |
| Application          | Streamlit     |
| Language             | Python        |
| Machine Learning     | Scikit-learn  |
| Classification       | Random Forest |
| Data Processing      | Pandas        |
| Numerical Processing | NumPy         |
| Authentication       | bcrypt        |
| Model Serialization  | Pickle        |
| Unique IDs           | UUID          |
| Dataset              | CSV           |

---

## 11. Getting Started

### 11.1 Prerequisites

Install:

* Python 3
* pip

### 11.2 Clone the Repository

```bash
git clone https://github.com/Tanm00018/AI-Powered-Healthcare-System.git
cd AI-Powered-Healthcare-System
```

### 11.3 Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn bcrypt tqdm
```

If a `requirements.txt` file is added to the repository, dependencies can instead be installed with:

```bash
pip install -r requirements.txt
```

### 11.4 Train the Model

From the project root:

```bash
python train_model.py
```

The training process loads the dataset, trains the Random Forest classifier, evaluates the model, and writes the serialized model to the model directory.

### 11.5 Run the Application

```bash
streamlit run app.py
```

The Streamlit interface will then be available through the local development server.

---

## 12. Engineering Design

### 12.1 Separation of Training and Inference

Model training is isolated from the application runtime.

```text
train_model.py
      ↓
Trained Model
      ↓
disease_predictor.pkl
      ↓
app.py
      ↓
Inference
```

This allows the application to load an already-trained model rather than retraining it for every session.

### 12.2 Role-Based Workflows

Instead of presenting a single generic interface, the application branches into role-specific workflows:

```text
                 Authentication
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Patient            Doctor
              │                 │
        Patient Portal     Doctor Portal
```

This keeps patient-facing and doctor-facing operations conceptually separate.

### 12.3 Stateful Application Design

Streamlit session state is used to maintain:

```text
User Accounts
Login State
Username
Role
Selected Medical Folder
```

This enables the application to maintain user context across Streamlit interactions.

### 12.4 Modular Functional Design

Although the application is contained in a single Python module, major workflows are separated into dedicated functions for authentication, patient operations, doctor operations, record rendering, and application initialization.

---

## 13. Applications

The architecture provides a foundation for applications such as:

* Symptom-based health assessment interfaces
* Personal medical-history systems
* Doctor-patient record management
* Healthcare information dashboards
* ML-assisted healthcare prototypes
* Educational healthcare AI systems
* Medical record organization tools

The machine-learning component can also serve as a foundation for experimenting with additional classification and decision-support techniques.

---

## 14. Future Development

The current architecture provides several natural directions for expansion:

* Persistent database-backed storage
* Secure production authentication
* Patient-doctor relationship management
* Model confidence and probability visualization
* More robust feature preprocessing
* Model comparison and hyperparameter optimization
* Cross-validation and expanded evaluation metrics
* Explainable disease predictions
* Patient history analytics
* Cloud deployment
* REST API integration
* Automated medical-report processing
* Advanced NLP-based symptom extraction
* Retrieval-based medical information assistance

---

## 15. Project Vision

The broader objective of the project is to explore how **machine learning can be integrated into practical healthcare software rather than being treated as an isolated prediction model**.

The system combines three layers:

```text
Healthcare Data
      +
Machine Learning
      +
Interactive Application
```

The result is a prototype healthcare platform where symptom-based prediction exists alongside patient records, doctor workflows, treatment information, and medical-history management.

The architecture provides a foundation for evolving the project toward a more comprehensive **AI-assisted healthcare information and decision-support platform**.

---

## 16. Built With

**Python · Streamlit · Scikit-learn · Random Forest · Pandas · NumPy · bcrypt**

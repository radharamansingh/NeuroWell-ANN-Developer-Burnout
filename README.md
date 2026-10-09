# NeuroWell 🧠
### ANN-Based Developer Wellness Risk Intelligence System

NeuroWell is a Deep Learning portfolio project that demonstrates how an Artificial Neural Network (ANN) can classify developer workload and lifestyle indicators into three simulated risk categories: **Low, Medium, and High**.

The project combines a TensorFlow/Keras neural network with an interactive Streamlit dashboard, allowing users to enter workload-related information and explore the model's predictions.

> **Disclaimer:** This is an educational prototype trained on synthetic data. It is not a clinically validated burnout detector, and its predictions should not be used for medical or employment decisions.

## 🎯 Project Objectives

- Implement a feedforward Artificial Neural Network for multiclass classification.
- Explore the effect of workload and lifestyle features on simulated risk labels.
- Apply feature scaling and evaluate model performance.
- Deploy the trained model through an interactive web interface.
- Demonstrate a complete Deep Learning workflow from data generation to inference.

## ✨ Key Features

- Synthetic dataset containing 10,000 records.
- Ten input features related to developer workload and lifestyle.
- ANN with two hidden layers and ReLU activation.
- Dropout regularization and Early Stopping.
- Adam optimizer and sparse categorical cross-entropy loss.
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
- Interactive Streamlit interface with adjustable inputs and class-probability visualization.
- Saved model and preprocessing artifacts for inference.

## 🛠️ Technology Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** Pandas, NumPy
- **Preprocessing and Evaluation:** Scikit-learn
- **Model Persistence:** Keras model format, Joblib
- **Web Application:** Streamlit

## 📊 Dataset

The dataset contains 10,000 synthetically generated records.

| Feature | Description |
|---|---|
| working_hours | Daily working hours |
| sleep_hours | Daily sleep duration |
| coding_hours | Daily coding hours |
| meetings_per_day | Number of meetings |
| breaks_per_day | Number of breaks |
| deadline_pressure | Simulated pressure score (1–10) |
| job_satisfaction | Simulated satisfaction score (1–10) |
| work_life_balance | Simulated balance score (1–10) |
| consecutive_working_days | Consecutive working days |
| tasks_completed | Daily completed tasks |

**Target variable:** `burnout_risk`

- `0` — Low
- `1` — Medium
- `2` — High

The target labels are generated using a predefined scoring formula and quantile-based thresholds. Consequently, the model learns patterns in this simulated labeling process rather than validated real-world burnout outcomes.

## 🧠 ANN Architecture

```text
10 Input Features
        ↓
Dense Layer — 64 Neurons (ReLU)
        ↓
Dropout — 20%
        ↓
Dense Layer — 32 Neurons (ReLU)
        ↓
Output Layer — 3 Neurons (Softmax)
        ↓
Low / Medium / High
```

**Training configuration**

- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Batch size: 32
- Maximum epochs: 40
- Early stopping: Monitors validation loss
- Train/test split: 80% / 20%
- Feature scaling: StandardScaler

## 📈 Model Evaluation

The model was evaluated on a held-out test set of 2,000 synthetic records.

| Metric | Result |
|---|---:|
| Test Accuracy | 77.25% |
| Macro Precision | 0.77 |
| Macro Recall | 0.77 |
| Macro F1-score | 0.77 |

### Class-wise F1-score

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Low | 0.81 | 0.83 | 0.82 |
| Medium | 0.67 | 0.67 | 0.67 |
| High | 0.85 | 0.82 | 0.83 |

These are results from one training run and should not be interpreted as evidence of real-world predictive performance.

## 🖥️ Application Workflow

1. Enter workload and lifestyle indicators in the Streamlit interface.
2. Apply the saved feature scaler.
3. Pass the scaled inputs to the trained ANN.
4. Display the predicted class and model probability distribution.

The displayed class probability is the model's estimated output for that class, not a calibrated probability that a person is experiencing burnout.

## 📁 Project Structure

```text
ANN_Developer_Burnout/
├── app/
│   ├── app.py
│   └── train_model.py
├── data/
│   ├── generate_dataset.py
│   └── developer_wellbeing.csv
├── models/
│   ├── neurowell_ann.keras
│   ├── scaler.pkl
│   ├── features.json
│   └── training_history.json
├── notebooks/
├── README.md
└── requirements.txt
```

## 🚀 Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ANN_Developer_Burnout
```

### 2. Install dependencies

Use a compatible Python environment and install the packages listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

### 3. Generate the synthetic dataset (optional)

```bash
python data/generate_dataset.py
```

### 4. Train the ANN (optional)

```bash
python app/train_model.py
```

Training regenerates the saved model and preprocessing artifacts.

### 5. Launch the application

```bash
python -m streamlit run app/app.py
```

Open the local URL displayed in the terminal.

## 🔮 Future Improvements

- Validate the approach using ethically collected, appropriately consented real-world data.
- Compare ANN performance with baseline models such as Logistic Regression and Random Forest.
- Perform hyperparameter tuning and cross-validation.
- Add training-history plots and more detailed evaluation visualizations.
- Improve model calibration and assess class-specific errors.

## 👨‍💻 Project Summary

NeuroWell demonstrates an end-to-end Deep Learning workflow involving synthetic data generation, preprocessing, ANN training, evaluation, model persistence, and Streamlit deployment.

The project is intended for learning and portfolio demonstration, not clinical assessment.

**Built with Python, TensorFlow/Keras, Scikit-learn, and Streamlit.**
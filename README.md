# Pneumonia Classification Using DenseNet-121, Fuzzy Logic, and Genetic Algorithm Optimization

**Course:** CSC 8810 – Computational Intelligence  
**University:** Georgia State University  
**Student:** Cherisha Killari Giribabu

## 1. Project Overview

This project investigates pneumonia classification using chest X-ray images through a combination of deep feature extraction, machine learning, fuzzy logic, and genetic algorithm optimization.

The implementation uses:

- **DenseNet-121:** Extracts 1024-dimensional features from chest X-ray images.
- **Logistic Regression:** Performs binary classification.
- **Gaussian Fuzzy Transformation:** Converts extracted features into fuzzy membership values.
- **Genetic Algorithm (GA):** Optimizes fuzzy membership parameters.

The project includes two experiments:

| Experiment | Description | Python File |
|---|---|---|
| Original Experiment | Original CSC 8810 implementation, including baseline, fuzzy, GA+Fuzzy, and random comparisons | `main.py` |
| Controlled Experiment | Additional comparison using fixed data splits, training-only scaling, and a shared held-out test set | `controlled_experiment.py` |

Both experiments are included in the **main branch** of this repository.

The original experiment is retained for comparison with the course project results. The controlled experiment provides a more consistent evaluation of the three trained classifiers.

---

## 2. Repository Structure

```text
CSC8810-Pneumonia-Classification/
│
├── main.py
├── controlled_experiment.py
├── demo.py
├── gradcam.py
├── requirements.txt
├── README.md
│
├── preprocessing/
│   └── preprocess.py
│
├── feature_extraction/
│   └── densenet_features.py
│
├── fuzzy_system/
│   └── fuzzy_layer.py
│
├── genetic_algorithm/
│   └── ga_optimizer.py
│
├── training/
│   └── train_classifier.py
│
├── evaluation/
│   └── evaluate.py
│
├── results/
│   ├── Original experiment figures
│   └── controlled/
│       ├── metrics.csv
│       ├── baseline_confusion.png
│       ├── baseline_roc.png
│       ├── fuzzy_confusion.png
│       ├── fuzzy_roc.png
│       ├── ga_fuzzy_confusion.png
│       ├── ga_fuzzy_roc.png
│       ├── ga_history.csv
│       ├── ga_best_validation.png
│       └── run_details.json
│
├── model_baseline.pkl
├── scaler.pkl
└── test img/
```

The chest X-ray dataset is not included in the repository because of its size.

---

## 3. Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)  
**Source:** Kaggle – Paul Mooney

Dataset link:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The experiments use the dataset's training subset containing **5,216 chest X-ray images**.

| Class | Images |
|---|---:|
| Normal | 1,341 |
| Pneumonia | 3,875 |
| **Total** | **5,216** |

### Dataset Setup

1. Download the dataset from Kaggle.
2. Extract the downloaded archive.
3. Create a folder named `dataset` inside the cloned repository.
4. Arrange the extracted files as shown below.

```text
CSC8810-Pneumonia-Classification/
└── dataset/
    └── chest_xray/
        └── train/
            ├── NORMAL/
            └── PNEUMONIA/
```

**Important:** Both experiments expect the images at:

```text
dataset/chest_xray/train
```

The dataset path must be correct before running either experiment.

---

## 4. System Requirements

The original project was developed and tested using **Python 3.10**.

The required libraries are listed in `requirements.txt`:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Joblib
- TensorFlow
- Seaborn

The repository specifies TensorFlow 2.14.0 and NumPy 1.26.4.

**Recommended environment:** Python 3.10 with a virtual environment.

A GPU can accelerate DenseNet-121 feature extraction, but the implementation can also run on a CPU. CPU execution may take considerably longer.

---

## 5. Installation and Setup

### Step 1 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/cherisha2003/CSC8810-Pneumonia-Classification.git
```

Navigate to the project directory:

```bash
cd CSC8810-Pneumonia-Classification
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 3 — Activate the Environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### Step 4 — Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The first execution may download pretrained ImageNet weights for DenseNet-121.

### Step 5 — Verify Dataset Placement

Confirm that the following folders exist:

```text
dataset/chest_xray/train/NORMAL
dataset/chest_xray/train/PNEUMONIA
```

Once the dataset and dependencies are ready, either experiment can be executed.

---

## 6. Experiment 1 — Original CSC 8810 Implementation

**File:** `main.py`

### Run the Original Experiment

From the repository root, execute:

```bash
python main.py
```

### What This Experiment Does

1. Loads and preprocesses chest X-ray images.
2. Resizes images to 224 × 224 pixels.
3. Extracts 1024-dimensional features using pretrained DenseNet-121.
4. Applies feature standardization.
5. Trains a Logistic Regression baseline.
6. Applies Gaussian fuzzy feature transformation.
7. Optimizes fuzzy parameters using a Genetic Algorithm.
8. Performs the implemented evaluation and cross-validation procedures.
9. Generates performance plots and saves the baseline model artifacts.

### Original Experiment Results

The original course experiment reported:

| Model | Reported Accuracy |
|---|---:|
| DenseNet-121 + Logistic Regression | 98.30% |
| Fuzzy + Logistic Regression | 95.40% |
| GA+Fuzzy + Logistic Regression | 94.20% |
| Random Baseline | 53.50% |

These values are retained from the original course experiment.

**Reproducibility note:** The original implementation uses randomized operations, and its model comparisons do not all use identical evaluation procedures. Consequently, a new execution may produce different numerical results.

### Original Results

Historical figures are stored in:

```text
results/
```

Examples include:

```text
results/ablation.png
results/roc.png
results/confusion.png
results/tsne.png
results/cv_fold_accuracies.png
```

The historical figures are retained from the original experiment. They should not be interpreted as outputs from the controlled experiment.

---

## 7. Experiment 2 — Controlled Comparison

**File:** `controlled_experiment.py`

This experiment was added to provide a more consistent comparison between the DenseNet baseline, Fuzzy, and GA+Fuzzy classifiers.

It is separate from the original experiment and does not replace its saved results or GUI model.

### Run the Controlled Experiment

From the repository root, execute:

```bash
python controlled_experiment.py
```

### Experimental Configuration

| Parameter | Configuration |
|---|---|
| Dataset | 5,216 chest X-ray images |
| Feature extractor | DenseNet-121 |
| Feature dimension | 1024 |
| Random seed | 42 |
| Training partition | 60% |
| Validation partition | 20% |
| Test partition | 20% |
| Data splitting | Stratified |
| Standardization | Scaler fitted on training data only |
| Classifier | Logistic Regression |
| GA population | 20 |
| GA generations | 20 |

### Controlled Experiment Procedure

1. Load the dataset and extract DenseNet-121 features.
2. Divide the data into stratified training, validation, and test partitions.
3. Fit the `StandardScaler` using the training partition only.
4. Train the baseline classifier.
5. Apply fuzzy transformation using fixed randomly initialized parameters.
6. Optimize fuzzy membership parameters using the Genetic Algorithm.
7. Use training and validation data for GA optimization without accessing the held-out test labels.
8. Train the final classifiers using the combined training and validation partitions.
9. Evaluate all three classifiers on the same held-out test partition.
10. Save evaluation metrics, ROC curves, confusion matrices, and GA optimization history.

### Controlled Experiment Results

The saved results in `results/controlled/metrics.csv` are:

| Model | Test Accuracy | Balanced Accuracy | ROC AUC |
|---|---:|---:|---:|
| DenseNet Baseline | 98.28% | 97.74% | 0.9985 |
| Fuzzy | 97.22% | 96.67% | 0.9954 |
| GA+Fuzzy | 96.55% | 95.73% | 0.9960 |

**Held-out test samples:** 1,044

These results correspond to the recorded controlled experiment and are reported to the displayed precision.

### Controlled Experiment Outputs

All controlled experiment results are saved separately in:

```text
results/controlled/
```

**Evaluation metrics:**

```text
metrics.csv
run_details.json
```

**Confusion matrices:**

```text
baseline_confusion.png
fuzzy_confusion.png
ga_fuzzy_confusion.png
```

**ROC curves:**

```text
baseline_roc.png
fuzzy_roc.png
ga_fuzzy_roc.png
```

**Genetic Algorithm results:**

```text
ga_history.csv
ga_best_validation.png
```

The controlled experiment does not overwrite the original figures or saved GUI model.

### Interpretation

The DenseNet-121 baseline achieved the highest accuracy in the controlled comparison.

The fuzzy and GA+Fuzzy approaches were evaluated to investigate whether feature transformation and genetic optimization improved classification performance.

In the recorded experiment, neither approach outperformed the baseline.

---

## 8. Running the GUI Demo

**File:** `demo.py`

The repository includes a GUI for testing individual chest X-ray images.

Run:

```bash
python demo.py
```

### How to Use

1. Launch the application.
2. Click **Upload X-ray Image**.
3. Select a JPG, JPEG, or PNG image.
4. View the predicted class and displayed confidence score.

Sample images are available in:

```text
test img/
```

### GUI Model

The GUI uses:

- DenseNet-121 feature extraction
- The saved feature scaler (`scaler.pkl`)
- The saved Logistic Regression baseline (`model_baseline.pkl`)

**Important:** The GUI uses the original baseline classifier. It does not use the controlled experiment's GA+Fuzzy model.

---

## 9. Reproducibility and Experimental Limitations

### Original Experiment

The original experiment contains randomized processes, including Genetic Algorithm optimization and train/test operations.

Exact results may vary between executions.

Some preprocessing and evaluation choices in the original implementation also limit direct comparisons between methods.

### Controlled Experiment

The controlled experiment addresses several of these limitations through:

- A fixed random seed.
- Stratified dataset partitioning.
- Training-only standardization.
- A shared held-out test set.
- Separate validation data for Genetic Algorithm optimization.

The saved controlled experiment metrics represent a completed run. Running the script again allows the experiment to be reproduced, although small differences may occur depending on the software environment and hardware.

### Additional Considerations

The experiments use the training subset of a publicly available chest X-ray dataset.

The results should not be interpreted as external clinical validation.

Further evaluation on independent datasets would be required to assess generalization.

---

## 10. Quick Execution Reference

| Task | Command or Location |
|---|---|
| Install dependencies | `python -m pip install -r requirements.txt` |
| Run original experiment | `python main.py` |
| Run controlled experiment | `python controlled_experiment.py` |
| Run GUI | `python demo.py` |
| View original figures | `results/` |
| View controlled figures | `results/controlled/` |
| View controlled metrics | `results/controlled/metrics.csv` |
| View experiment configuration | `results/controlled/run_details.json` |

**For reviewing the CSC 8901 Perspectives in CS paper, both experiments are relevant.** The original experiment documents the initial implementation and results, while the controlled experiment provides a shared-split evaluation of the trained classifiers.

---

## 11. Disclaimer

This project was developed for academic and research purposes only.

It is not intended for clinical diagnosis, medical decision-making, or deployment in healthcare settings.

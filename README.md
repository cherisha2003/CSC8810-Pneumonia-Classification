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

The repository contains two experiments:

| Experiment | Description | Python File |
|---|---|---|
| Original Experiment | Original CSC 8810 implementation, including baseline, fuzzy, GA+Fuzzy, and random comparisons | `main.py` |
| Controlled Experiment | Additional comparison using fixed data splits, training-only scaling, and a shared held-out test set | `controlled_experiment.py` |

Both experiments are available on the `main` branch.

The original experiment is retained to document the initial course project. The controlled experiment provides a more consistent comparison of the three trained classifiers.

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

Dataset download:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The experiments use the dataset's training subset containing **5,216 chest X-ray images**.

| Class | Images |
|---|---:|
| Normal | 1,341 |
| Pneumonia | 3,875 |
| **Total** | **5,216** |

### Dataset Setup

1. Download the dataset from Kaggle using the link above.
2. Extract the downloaded archive.
3. Create a folder named `dataset` inside the cloned repository.
4. Arrange the extracted files according to the following structure:

```text
CSC8810-Pneumonia-Classification/
└── dataset/
    └── chest_xray/
        └── train/
            ├── NORMAL/
            └── PNEUMONIA/
```

Both experiments expect the dataset at:

```text
dataset/chest_xray/train
```

**Important:**

- The dataset must be downloaded separately.
- Installing `requirements.txt` does not download the dataset.
- No additional Python package is required when downloading and extracting the dataset manually.
- The dataset folders must be correctly placed before running either experiment.

---

## 4. System Requirements

### Python Version

**Python 3.10 is required.**

The project uses TensorFlow 2.14.0 and NumPy 1.26.4, which are specified in `requirements.txt`.

These dependencies were successfully installed and imported using Python 3.10.22 in GitHub Codespaces.

Using newer Python versions, such as Python 3.14, may cause dependency installation failures.

### Required Libraries

The required Python packages are listed in `requirements.txt`:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Joblib
- TensorFlow
- Seaborn

### Hardware

A GPU is recommended for faster DenseNet-121 feature extraction, but the experiments can also run on a CPU.

CPU execution may take considerably longer because the project processes thousands of chest X-ray images.

Ensure that sufficient memory and disk space are available for the dataset, extracted features, and pretrained model weights.

---

## 5. Installation and Setup

### Step 1 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/cherisha2003/CSC8810-Pneumonia-Classification.git
```

Navigate to the repository:

```bash
cd CSC8810-Pneumonia-Classification
```

### Step 2 — Verify Python Version

Check your Python version:

```bash
python3 --version
```

The recommended version is Python 3.10.x.

If Python 3.10 is installed, create a virtual environment:

```bash
python3.10 -m venv venv
```

If your system uses `python` for Python 3.10, you can instead run:

```bash
python -m venv venv
```

### Step 3 — Activate the Virtual Environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

Verify the active Python version:

```bash
python --version
```

It should display Python 3.10.x.

### Step 4 — Install Dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

This installs the required Python packages, including TensorFlow, NumPy, Scikit-learn, and OpenCV.

### Additional Linux Dependencies

Some Linux environments require additional system libraries for OpenCV.

During testing in GitHub Codespaces, importing OpenCV initially failed with:

```text
ImportError: libGL.so.1: cannot open shared object file
```

The issue was resolved by installing:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

This step is intended for Ubuntu/Debian-based Linux environments and is not required on every operating system.

### Step 5 — Verify Package Installation

Run:

```bash
python -c "import tensorflow as tf; import numpy as np; import sklearn; import cv2; print('TensorFlow:', tf.__version__); print('NumPy:', np.__version__); print('Scikit-learn:', sklearn.__version__); print('OpenCV:', cv2.__version__); print('ALL IMPORTS SUCCESSFUL')"
```

A successfully tested configuration produced:

```text
TensorFlow: 2.14.0
NumPy: 1.26.4
Scikit-learn: 1.7.2
OpenCV: 4.11.0
ALL IMPORTS SUCCESSFUL
```

TensorFlow may display CUDA-related warnings on CPU-only machines. These warnings do not necessarily prevent CPU execution.

### Step 6 — Verify Dataset Placement

Ensure the following folders exist:

```text
dataset/chest_xray/train/NORMAL
dataset/chest_xray/train/PNEUMONIA
```

Once the dataset and dependencies are ready, either experiment can be executed.

---

## 6. Alternative Setup for GitHub Codespaces

GitHub Codespaces may use a newer Python version by default.

If Python 3.10 is unavailable, it can be installed using `uv`.

From the repository root, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add `uv` to the current terminal session:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Install Python 3.10:

```bash
uv python install 3.10
```

Create a virtual environment:

```bash
uv venv --python 3.10 .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Verify the Python version:

```bash
python --version
```

Install the project's requirements:

```bash
uv pip install -r requirements.txt
```

If OpenCV reports a missing `libGL.so.1` library, install the Linux dependencies described in Section 5.

**Note:** The Codespaces setup is an alternative to the standard virtual environment instructions. It is not necessary to perform both methods.

---

## 7. Experiment 1 — Original CSC 8810 Implementation

**File:** `main.py`

### Run the Original Experiment

From the repository root, execute:

```bash
python main.py
```

### Experiment Procedure

The original implementation:

1. Loads chest X-ray images from the dataset.
2. Resizes images to 224 × 224 pixels.
3. Extracts 1024-dimensional features using pretrained DenseNet-121.
4. Standardizes the extracted features.
5. Trains a Logistic Regression baseline classifier.
6. Applies Gaussian fuzzy feature transformation.
7. Uses a Genetic Algorithm to optimize fuzzy parameters.
8. Evaluates the classifiers using the implemented procedures.
9. Generates performance visualizations.
10. Saves the baseline model and scaler.

### Original Experiment Results

The original course experiment reported:

| Model | Reported Accuracy |
|---|---:|
| DenseNet-121 + Logistic Regression | 98.30% |
| Fuzzy + Logistic Regression | 95.40% |
| GA+Fuzzy + Logistic Regression | 94.20% |
| Random Baseline | 53.50% |

These values are retained from the original course project.

**Reproducibility note:** The original implementation uses randomized operations and different evaluation procedures for some comparisons. Consequently, a new execution may produce different results.

### Original Experiment Outputs

The original experiment generates figures in:

```text
results/
```

Examples include:

```text
results/tsne_embeddings.png
results/ga_fitness_curve.png
results/ga_population_diversity.png
results/cv_fold_accuracies.png
```

It also saves:

```text
model_baseline.pkl
scaler.pkl
```

These model artifacts are used by the GUI demonstration.

---

## 8. Experiment 2 — Controlled Comparison

**File:** `controlled_experiment.py`

The controlled experiment was added to provide a more consistent comparison between the DenseNet baseline, Fuzzy, and GA+Fuzzy classifiers.

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
7. Use training and validation data for GA optimization without accessing held-out test labels.
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

These values correspond to the recorded controlled experiment and are reported to the displayed precision.

### Controlled Experiment Outputs

The controlled experiment saves its outputs in:

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

The controlled experiment does not overwrite the original experiment's figures or saved GUI model.

### Interpretation

The DenseNet-121 baseline achieved the highest accuracy in the controlled comparison.

The fuzzy and GA+Fuzzy approaches were evaluated to investigate whether feature transformation and genetic optimization improved classification performance.

In the recorded experiment, neither approach outperformed the baseline.

---

## 9. Running the GUI Demo

**File:** `demo.py`

The repository includes a GUI for testing individual chest X-ray images.

### Run the GUI

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

- DenseNet-121 feature extraction.
- The saved feature scaler (`scaler.pkl`).
- The saved Logistic Regression baseline (`model_baseline.pkl`).

**Important:** The GUI uses the original baseline classifier. It does not use the controlled experiment's GA+Fuzzy model.

### GUI Environment Requirement

The GUI requires a graphical desktop environment.

It may not display in a headless GitHub Codespaces terminal without additional display configuration.

This does not affect the command-line experiments.

---

## 10. Reproducibility and Experimental Limitations

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

The saved controlled experiment metrics represent a completed run.

Running the script again may produce small numerical differences depending on the software environment and hardware.

### Environment Testing

The following setup was successfully tested in GitHub Codespaces:

| Component | Tested Version | Status |
|---|---|---|
| Python | 3.10.22 | Passed |
| TensorFlow | 2.14.0 | Passed |
| NumPy | 1.26.4 | Passed |
| Scikit-learn | 1.7.2 | Passed |
| OpenCV | 4.11.0 | Passed |
| Requirements installation | `requirements.txt` | Passed |
| Python library imports | All key libraries | Passed |

**Testing status:** The dataset setup and complete execution of `main.py` and `controlled_experiment.py` have not yet been verified in the fresh Codespaces environment.

### Additional Considerations

The experiments use the training subset of a publicly available chest X-ray dataset.

The results should not be interpreted as external clinical validation.

Further evaluation on independent datasets would be required to assess generalization.

---

## 11. Quick Execution Reference

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

For reviewing the CSC 8901 Perspectives in CS paper, both experiments are relevant.

The original experiment documents the initial implementation and results, while the controlled experiment provides a shared-split evaluation of the trained classifiers.

---

## 12. Disclaimer

This project was developed for academic and research purposes only.

It is not intended for clinical diagnosis, medical decision-making, or deployment in healthcare settings.

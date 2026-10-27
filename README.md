# CSC 8810 - Pneumonia Classification

Course project for CSC 8810 Computational Intelligence at Georgia State University.

This project evaluates pneumonia classification from chest X-ray images using DenseNet-121 feature extraction, Logistic Regression, fuzzy feature transformation, and Genetic Algorithm optimization.

## Project Files

- `main.py` - runs the complete experiment
- `demo.py` - GUI for testing chest X-ray images
- `gradcam.py` - Grad-CAM visualization code
- `preprocessing/` - image preprocessing
- `feature_extraction/` - DenseNet-121 feature extraction
- `fuzzy_system/` - fuzzy transformation
- `genetic_algorithm/` - Genetic Algorithm optimization
- `training/` - classifier training
- `evaluation/` - evaluation functions
- `results/` - generated plots and experiment results
- `test img/` - sample images for testing the GUI
- `model_baseline.pkl` - saved Logistic Regression baseline model
- `scaler.pkl` - saved feature scaler

## Dataset

This project uses the Chest X-Ray Images (Pneumonia) dataset available on Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset is not included in this repository because of its size.

After downloading and extracting the dataset, place it inside the project using the following structure:

```text
dataset/
└── chest_xray/
    └── train/
        ├── NORMAL/
        └── PNEUMONIA/
```

The program expects the training dataset at:

```text
dataset/chest_xray/train
```

The training portion contains 5,216 chest X-ray images.

## Requirements

The project was tested using Python 3.10.

The required Python packages are listed in `requirements.txt`.

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The first execution may download the pretrained ImageNet weights used by DenseNet-121.

## Run the Full Experiment

Make sure the dataset is placed in the expected directory, then run:

```bash
python main.py
```

The program performs preprocessing, DenseNet-121 feature extraction, Logistic Regression baseline classification, fuzzy feature transformation, Genetic Algorithm optimization, cross-validation, and evaluation.

DenseNet features are extracted in batches to reduce memory usage.

The full experiment processes all 5,216 training images and may take some time to complete, especially when running on CPU.

Generated graphs are saved in the `results/` folder.

The program also saves the baseline model and feature scaler for use by the GUI.

## Run the GUI Demo

The repository includes the saved baseline model and scaler, so the GUI can be tested without rerunning the full experiment.

Run:

```bash
python demo.py
```

Click **Upload X-ray Image** and select a JPG, JPEG, or PNG chest X-ray image.

The application displays the predicted class and confidence score.

Sample images are available in the `test img/` folder.

## Demo Model

The GUI uses DenseNet-121 for feature extraction together with the saved feature scaler and Logistic Regression baseline classifier.

The baseline classifier is used because it achieved the strongest performance among the evaluated approaches.

## Reproducibility Note

The Genetic Algorithm and some train/test operations involve randomized processes. Exact numerical results may therefore vary slightly between executions.

The figures in the `results/` directory represent the results retained from the original course project.

## Important Note

This project was developed for academic purposes only and is not intended for clinical diagnosis or medical decision-making.

## Controlled comparison (new, recommended for CSC 8901 review)

Run `python controlled_experiment.py` from the repository root after placing the Kaggle dataset as described above. This is **separate from the original course experiment** (`python main.py`) and does not overwrite the original plots or saved GUI model.

The controlled experiment uses a fixed seed (42), stratified 60% training / 20% validation / 20% held-out test partition, and fits `StandardScaler` on training data only. All three classifiers are tested on the **same untouched test indices**. GA fitness uses training and validation data only, remembers the best-ever candidate, and does not access test labels. The final classifiers are fitted on combined train+validation data using the chosen transformation parameters.

New outputs are saved under `results/controlled/`:

- `metrics.csv` — held-out accuracy, balanced accuracy, AUC for each model
- `baseline_confusion.png`, `fuzzy_confusion.png`, `ga_fuzzy_confusion.png`
- `baseline_roc.png`, `fuzzy_roc.png`, `ga_fuzzy_roc.png`
- `ga_history.csv`, `ga_best_validation.png`, `run_details.json`

**Original figure naming:** `main.py` generates `results/ablation_comparison.png`, `results/roc_curve.png`, and `results/confusion_matrix.png`, whereas historical files retained from the course project are `results/ablation.png`, `results/roc.png`, and `results/confusion.png`. These historical images are not claimed to have been regenerated by the controlled script. Only figures under `results/controlled/` are attributable to the new, fixed-split experiment.

**Do not cite controlled metrics before running this script.** The Kaggle dataset is intentionally excluded from this repository. A full reproduction requires downloading it and the pretrained ImageNet weights. CPU runs may be lengthy. The existing GUI uses the original saved baseline artifacts; the controlled script does not replace them.

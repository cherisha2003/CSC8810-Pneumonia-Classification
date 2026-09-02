# CSC 8810 - Pneumonia Classification

Course project for CSC 8810 Computational Intelligence at Georgia State University.

This project evaluates pneumonia classification from chest X-ray images using DenseNet-121 feature extraction, Logistic Regression, fuzzy feature transformation, and Genetic Algorithm optimization.

## Project Files

- main.py - runs the complete experiment
- demo.py - GUI for testing chest X-ray images
- gradcam.py - Grad-CAM visualization code
- preprocessing/ - image preprocessing
- feature_extraction/ - DenseNet-121 feature extraction
- fuzzy_system/ - fuzzy transformation
- genetic_algorithm/ - GA optimization
- training/ - classifier training
- evaluation/ - evaluation functions
- results/ - generated plots and results
- test img/ - sample images for the GUI

## Dataset

The dataset is not included in this repository because of its size.

Place the downloaded dataset in this structure:

dataset/
  chest_xray/
    train/
      NORMAL/
      PNEUMONIA/

The program expects the training dataset at:

dataset/chest_xray/train

## Installation

Create a virtual environment:

python3 -m venv venv

Activate it on macOS/Linux:

source venv/bin/activate

Install the required packages:

pip install -r requirements.txt

## Run the Full Experiment

Run:

python main.py

This executes preprocessing, DenseNet-121 feature extraction, classification, fuzzy feature transformation, Genetic Algorithm optimization, cross-validation, and evaluation.

Generated graphs are saved in the results folder.

## Run the GUI Demo

Run:

python demo.py

Click "Upload X-ray Image" and select a JPG, JPEG, or PNG image.

The application displays the predicted class and confidence score.

Sample images are included in the test img folder.

## Demo Model

The GUI uses the saved DenseNet-121 feature extractor with the Logistic Regression baseline classifier.

The baseline model is used because it achieved the strongest performance in the experiments.

## Important Note

This project was developed for academic purposes and is not intended for clinical diagnosis.

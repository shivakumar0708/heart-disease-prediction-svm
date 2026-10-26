# Heart Disease Prediction Using SVM

A machine-learning academic project that explores heart-disease data, preprocesses patient-related features, compares Support Vector Machine (SVM) kernels, and demonstrates predictions through a Gradio interface.

> **Important:** This is an educational project, not a medical diagnostic tool. Do not use its predictions for medical decisions.

## Project files

- `Heart_Disease_Prediction_SVM.ipynb` — main Google Colab/Jupyter notebook containing data exploration, preprocessing, SVM training, evaluation, model saving, and the Gradio interface.
- `heart_disease_augmented_practice.csv` — dataset used by the notebook.
- `Heart_Disease_Prediction_SVM_Review.pptx` — project review presentation (included if supplied).
- `website_screenshot.png` — screenshot of the Gradio interface (included if supplied).

## Models compared

The notebook compares three SVM kernels:

1. Linear SVM
2. Polynomial SVM
3. RBF SVM

It evaluates the models using accuracy, precision, recall, F1-score, and confusion matrices, then selects the model with the best F1-score in that run.

## Run in Google Colab

1. Upload `Heart_Disease_Prediction_SVM.ipynb` to Google Colab, or open it from your GitHub repository.
2. Upload `heart_disease_augmented_practice.csv` when the notebook asks for the dataset.
3. Run the notebook cells in order.
4. The notebook trains and compares the SVM models, saves `heart_disease_svm.pkl` and `heart_disease_scaler.pkl`, and launches a Gradio prediction interface in the final cell.

The notebook uses Google Colab's `files.upload()` in its dataset-loading step. If you run it locally in Jupyter, adjust that cell to load the CSV directly, for example:

```python
import pandas as pd
df = pd.read_csv("heart_disease_augmented_practice.csv")
```

## Requirements

The main libraries used by the notebook are listed in `requirements.txt`. In Google Colab, most scientific Python packages are already available; the notebook installs Gradio in its final setup steps.

For a local environment, install dependencies with:

```bash
pip install -r requirements.txt
```

Then open the notebook in Jupyter or VS Code and run the cells in order. The notebook currently contains Colab-specific upload code and notebook magic commands, so those cells may need small adjustments for a purely local run.

## Data and responsible use

The dataset is included for this academic project. Confirm you have permission to redistribute it before publishing this repository publicly. The model's performance on one train/test split does not establish clinical reliability or generalization to real patients.

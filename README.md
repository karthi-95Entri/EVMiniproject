# Electric Vehicle Population Analysis and EV Type Classification

**Student:** Karthikeyan P  
**Batch:** DSB04  
**Type:** Data Science Mini Project (Supervised Learning – Classification)

## Overview

Electric vehicles (EVs) are either **Battery Electric Vehicles (BEV)** or **Plug-in Hybrid Electric Vehicles (PHEV)**. This project explores an EV population dataset, cleans it, visualizes the main patterns, and trains machine learning models to predict whether a vehicle record is a BEV or a PHEV.

> **Scope note:** The dataset contains vehicle population records. It is not a measure of EV sales, charging activity, or future demand, so the results describe this dataset only.

## Objectives

- Understand the dataset structure and data quality
- Handle missing values and check for duplicate records
- Perform exploratory data analysis (EDA) with summary statistics and charts
- Compare EV types, manufacturers, model years, and electric range
- Train and compare multiple classification models
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrices
- Discuss limitations and future improvements

## Dataset

| Item | Detail |
|---|---|
| Original file | 256,902 rows × 11 columns |
| Source | [Electric Vehicle Population Data – data.gov](https://catalog.data.gov/dataset/electric-vehicle-population-data) |
| Data file used | [GitHub CSV](https://raw.githubusercontent.com/karthi-95Entri/EV-Dataset/refs/heads/main/EV%20Vehicle%20Population%20data.csv) |
| Filter applied | Only Washington (`State == "WA"`) records kept (256,492 rows); the `State` column is then dropped |
| Target variable | `Electric Vehicle Type` (BEV or PHEV) |

### Features

| Column | Type | Description |
|---|---|---|
| County, City | Categorical | Location of the vehicle |
| Postal Code, Legislative District | Numeric identifiers | Location codes |
| Model Year | Numeric | Vehicle model year |
| Make | Categorical | Manufacturer |
| Model | Categorical | Vehicle model |
| Electric Range | Numeric | Electric range recorded in the dataset |
| DOL Vehicle ID | Identifier | Excluded from modelling |
| Electric Vehicle Type | Categorical (target) | BEV or PHEV |

## Project Workflow

1. **Problem definition** – business problem, objectives, ML problem type
2. **Dataset understanding** – source, columns, target
3. **EDA** – overview, missing values and duplicates, correlation, 8 visualizations
4. **Preprocessing** – cleaning, duplicates, outlier check (IQR), skewness check, encoding, scaling, stratified train-test split
5. **Feature engineering** – final features: `Model Year`, `Electric Range`, `Make`, `Model`
6. **Model building** – Logistic Regression, Decision Tree, Random Forest, Linear SVM (all inside scikit-learn pipelines)
7. **Evaluation and comparison** – metrics, confusion matrices, best-model selection, overfitting check
8. **Conclusion and business impact**

## Key Steps and Findings

- **Missing values:** numeric columns filled with the median, categorical columns with the mode. No duplicate rows were found.
- **Electric Range outliers:** Q1 = 0, Q3 = 32, IQR upper bound = 80; 37,167 potential outliers were flagged but **not removed**, since high values may be valid.
- **Skewness:** Electric Range skewness is about 2.26. A log transform is shown only as a diagnostic; the original values are used for modelling.
- **Encoding and scaling:** one-hot encoding for categorical features and standardization for numeric features, both inside the pipeline so the preprocessing is fitted on training data only.
- **Sampling:** the classifier uses a stratified sample of up to 30,000 rows with an 80/20 train-test split (`random_state=42`).

## Model Results

Weighted metrics on the test set, taken from the notebook output:

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | 0.9887 | 0.9888 |
| Decision Tree | 0.9999 | 0.9999 |
| Random Forest | 0.9999 | 0.9999 |
| Linear SVM | 0.9891 | 0.9893 |

**Best model by weighted F1-score: Decision Tree** (training accuracy 1.0, test accuracy 0.9999, train-test gap about 0.0001).

## Charts

All 16 charts from the notebook are collected in **`EV_Project_Charts.docx`**:

1. Correlation heatmap (Model Year vs Electric Range)
2. EV type distribution (bar)
3. Top 10 manufacturers (bar)
4. EV records by model year (line)
5. Electric range distribution (histogram)
6. Electric range by EV type (box plot)
7. Top 10 counties (bar)
8. Top 10 vehicle models (bar)
9. EV type counts for top manufacturers (bar)
10. Electric range box plot (outlier detection)
11. Log1p-transformed electric range (skewness diagnostic)
12. Confusion matrix – Logistic Regression
13. Confusion matrix – Decision Tree
14. Confusion matrix – Random Forest
15. Confusion matrix – Linear SVM
16. Confusion matrix – best model

## Limitations and Future Work

- The near-perfect scores of the tree models suggest the features (`Make`, `Model`, `Electric Range`) almost determine the EV type, so the task is easy. This is likely partly a form of target leakage, and the result may not generalize to other vehicle sources.
- The dataset only covers Washington State vehicle records, not sales or usage.
- Zero values in Electric Range may be missing data rather than a true zero range.
- Possible improvements: cross-validation, hyperparameter tuning, `SelectKBest` inside the pipeline, trying features without `Electric Range`, and testing on other states or years.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn (Google Colab / Jupyter Notebook)

## How to Run

1. Open `EV_Vehicle_Population_Mini_Project.ipynb` in Google Colab or Jupyter.
2. Install the libraries if needed: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Run all cells from top to bottom. The dataset loads directly from the GitHub URL, so no manual upload is needed.

## Files

| File | Description |
|---|---|
| `EV_Vehicle_Population_Mini_Project.ipynb` | Full analysis and modelling notebook |
| `EV_Project_Charts.docx` | All chart outputs in a Word document |
| `README.md` | Project documentation (this file) |

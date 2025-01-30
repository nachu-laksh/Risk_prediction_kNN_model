
# Wildfire Risk Prediction for California Homes

## Overview

Given the increasing severity of wildfires in California, understanding which properties are most vulnerable to fire damage is crucial for policymakers, insurance companies, and homeowners. This project develops a predictive model to estimate the likelihood of homes sustaining significant damage (>50%) based on observable property characteristics.

Using **k-Nearest Neighbors (kNN) and Naïve Bayes classifiers**, this project analyzes how factors like construction materials, assessed value, and location influence wildfire damage. The study also provides a geospatial visualization of at-risk properties.

## Objectives

- Develop a **predictive model** to classify homes based on their risk of severe fire damage.
- Compare **kNN** and **Naïve Bayes** classifiers to determine the best-performing model.
- Visualize property risk using **scatter plots of latitude and longitude**.
- Offer insights to improve wildfire resilience in high-risk areas.

## Data

The dataset includes property attributes such as:
- **Structural characteristics**: Roof type, exterior siding, year built.
- **Financial information**: Assessed property value.
- **Geographic data**: Latitude, longitude, county.
- **Wildfire impact**: Damage classification (used as the target variable).

## Methodology

### 1. Data Cleaning & Feature Selection
- Handled missing values using domain-based imputation.
- Selected key features using **Mutual Information (MI)** analysis.
- Standardized categorical variables with **one-hot encoding**.

### 2. Model Training & Evaluation
- Used **k-fold cross-validation** to tune hyperparameters.
- Compared **Naïve Bayes** and **kNN (optimal k = 18)** based on accuracy and recall.
- Evaluated models using **confusion matrices and classification reports**.

### 3. Geospatial Analysis
- Visualized **damage distribution by county**.
- Visualized **damage distribution by roof and exterior siding materials**.
- Plotted high-risk homes on a **latitude-longitude scatter plot**.
  

## Key Findings

| Metric                | kNN  | Naïve Bayes | Explanation |
|----------------------|------|-------------|-------------|
| **Accuracy**        | 87%  | 77%        | Percentage of all correct predictions (damaged and non-damaged). |
| **Precision (Damage >50%)** | 89%  | 88%        | When the model predicts a home is damaged, this is the percentage that was actually damaged. |
| **Recall (Damage >50%)** | 89%  | 69%        | Out of all truly damaged homes, the percentage correctly identified as damaged. |
| **F1-score**        | 89%  | 77%        | Harmonic mean of precision and recall for balanced evaluation. |

- **kNN outperformed Naïve Bayes**, achieving **higher accuracy and recall for high-damage properties**.
- **False negatives (missed damaged homes)**: kNN model missed **1,597** truly damaged homes (11% miss rate).
- Counties with the highest destruction rates: **Mono, Lake, and San Luis Obispo**.


## Files

- **`Wildfire_risk_prediction.py`** – Python script for data processing, model training, and visualization.
- **`Wildfire_risk_report.doc`** – Summary report explaining findings for a non-technical audience.

## Recommendations

- **Improved Data**: Incorporate vegetation, fire history, and weather data for better predictions.
- **Threshold Tuning**: Adjust classification thresholds to minimize false negatives.
- **Policy Implications**: Encourage fire-resistant building materials in high-risk areas.


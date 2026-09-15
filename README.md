# Machine Learning Projects — Spotify Song Analysis

This repository contains coursework completed for the *Machine Learning for Data Analytics* module. The project addresses two machine learning problems using a Spotify song dataset: predicting a song's **popularity score** (regression) and predicting its **top genre** (classification).

## Overview

Music popularity and genre classification are influenced by a combination of audio characteristics. This project explores both problems using supervised machine learning techniques, comparing multiple models to identify the best-performing approach for each task.

## Tools & Libraries

- **Python** — core programming language
- **pandas, NumPy** — data manipulation and analysis
- **Matplotlib, Seaborn** — data visualization
- **scikit-learn** — model building, preprocessing, evaluation
- **XGBoost** — gradient boosting regression model
- **category_encoders** — categorical feature encoding
- **Jupyter Notebook** — development environment

## Input

The dataset (`CS98XClassificationTrain/Test.csv`, `CS98XRegressionTrain/Test.csv`) contains the following audio features for each song:

- `bpm` — tempo (beats per minute)
- `nrgy` — energy
- `dnce` — danceability
- `dB` — loudness
- `live` — liveness
- `val` — valence
- `dur` — duration
- `acous` — acousticness
- `spch` — speechiness
- `year`, `artist`, `title`, `top genre`

**Regression target:** `pop` (popularity score, 0–100)
**Classification target:** `top genre` (consolidated into broader genre categories)

## Output

- **Classification:** predicted genre category for each song (e.g. pop, rock, adult standards, other)
- **Regression:** predicted popularity score for each song
- Prediction files: `stacking_predictions.csv`, `random_forest_predictions.csv`, `xgboost_predictions.csv`
- Saved trained models: `decision_tree.pkl`, `svc_model.pkl`, `random_forest.pkl`, `rf_model.pkl`, `xgb_model.pkl`, `stacking_model.pkl`, `scaler.pkl`, `encoder.pkl`, `label_encoder.pkl`

## Repository Contents

| File | Description |
|---|---|
| `Team SG Classification (2).ipynb` | Genre classification notebook (Decision Tree, SVC, Random Forest) |
| `Team SG Regression (2) (1) (1).ipynb` | Popularity regression notebook (Random Forest, XGBoost, Stacking Ensemble) |
| `Team SG Report (1).pdf` | Full project report: methodology, EDA, model justification, and results |
| `Team SG final prediction.pdf` | Final prediction outputs |
| `Team SG (1).zip` | Supporting data and scripts |
| `machine learning projects -Copy1.ipynb` | Earlier standalone ML notebook |

## Methodology

**Preprocessing**
- Missing values filled using median (numerical) and mode (categorical)
- Outliers identified and treated using the interquartile range (IQR)
- Categorical features one-hot encoded
- Numerical features scaled using `StandardScaler`

**Classification models:** Decision Tree, Support Vector Classifier (SVC), Random Forest — evaluated using accuracy

**Regression models:** Random Forest, XGBoost, and a Stacking Ensemble (Random Forest + XGBoost, hyperparameter-tuned via `GridSearchCV`) — evaluated using RMSE

## Results

**Classification — Accuracy**

| Model | Accuracy |
|---|---|
| Decision Tree | 0.534 |
| SVC | 0.557 |
| Random Forest | 0.591 |

**Regression — RMSE**

| Model | Train | Validation |
|---|---|---|
| Random Forest | 5.39 | 10.60 |
| XGBoost | 5.33 | 10.89 |
| Stacking Ensemble | 5.40 | 10.70 |

Kaggle test score (regression): **7.778**

## Key Findings

- Loudness, danceability, and energy showed the strongest correlations with song popularity.
- Acousticness and loudness were the most influential features for genre classification.
- Random Forest performed best for classification; XGBoost generalized best for regression, with the Stacking Ensemble close behind.

## How to Run

1. Clone the repository and extract `Team SG (1).zip`
2. Install dependencies: pip install pandas numpy scikit-learn xgboost category_encoders matplotlib seaborn
3. Open the `classification & regression .ipynb` notebooks in Jupyter Notebook or Google Colab and run all cells in order

## Report

See `Team SG Report (1).pdf` for the complete methodology, exploratory data analysis, and detailed results.

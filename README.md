# Retweet Prediction Model

This project tests different models to predict the number of retweets a tweet will receive based on various features. The model uses an ensemble approach combining Random Forest, XGBoost, and Ridge Regression.

## Overview

The project analyzes tweet data to predict retweet counts using features such as:
* Text content (processed using TF-IDF and SVD)
* User metrics (followers, following, status counts)
* Tweet metadata (URLs, hashtags, verification status)
* Temporal features (hour of day, day of week)

## Project Structure

### 1. Data Exploration
* Dataset shape and basic statistics
* Feature distributions and correlations
* User metrics distribution
* Temporal patterns analysis
* Visualization of key relationships

### 2. Data Preprocessing
* Text vectorization using TF-IDF
* Dimensionality reduction using TruncatedSVD
* Feature engineering (temporal features, URL counts, etc.)
* Data scaling and normalization

### 3. Model Development
* Linear Regression (baseline)
* Ridge Regression
* Random Forest (GPU-accelerated)
* XGBoost
* Ensemble methods:
  * Weighted average
  * Stacking

### 4. Model Evaluation
* RMSE (Root Mean Square Error)
* MAE (Mean Absolute Error)
* R² Score
* Feature importance analysis
* Model comparison

## Key Features

* GPU acceleration using NVIDIA's cuML library
* Feature importance analysis
* Ensemble modeling
* Cross-validation
* Hyperparameter tuning

## Usage

1. Ensure all required libraries are installed
2. Make sure you download all the files in the repository
3. Run the notebook
4. Results will be saved in 'new_predictions.csv'

## Model Files

The following model files are saved during execution:
* `meta_model.pkl` (Ridge Regression)
* `rf_model.pkl`
* `xgb_model.pkl`
* `ridge_model.pkl`

## Performance

The project implements multiple models and combines them using ensemble techniques to achieve optimal performance. The stacking ensemble outperforms individual models.

![Performance Comparison](performance_comparison.png)

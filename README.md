# Binary Prediction of Smoker Status using Bio-Signals

## Project Overview

This project tackles a significant public health challenge: identifying an individual's smoking status based on non-invasive bio-signals. Using a comprehensive dataset of biological and physical measurements, this analysis systematically explores, evaluates, and compares various machine learning models to build a highly accurate classifier. The ultimate goal is to identify the most effective predictive model and the most influential bio-signals related to smoking.

The project demonstrates a full data science workflow, from exploratory data analysis and feature engineering to model training, hyperparameter tuning, and performance comparison.

## Table of Contents

- [Objective](#objective)
- [Key Features](#key-features)
- [Technologies & Libraries](#technologies--libraries)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run the Analysis](#how-to-run-the-analysis)
- [Modeling Approach](#modeling-approach)
- [Results & Key Findings](#results--key-findings)
- [Future Work](#future-work)
- [Contact](#contact)

## Objective

To develop a robust machine learning model that accurately classifies an individual as a smoker or non-smoker using a diverse set of 23 biological and physical markers, such as hemoglobin, height, weight, and cholesterol levels.

## Key Features

-   **Comprehensive EDA:** In-depth exploratory data analysis to uncover relationships between predictors and smoking status, including identifying multicollinearity with correlation heatmaps.
-   **Model Comparison:** Rigorous evaluation and comparison of five different classification models: Logistic Regression, Ridge Regression, Random Forest, XGBoost, and a Neural Network.
-   **Advanced Modeling:** Implementation of state-of-the-art ensemble methods (Random Forest and XGBoost) and deep learning (Neural Network) for superior predictive accuracy.
-   **Performance-Driven Results:** The XGBoost model achieved the highest performance, with a final **AUC score of 0.881**.
-   **Feature Importance Analysis:** Identification of the most critical bio-signals for predicting smoking status, with **hemoglobin, height, and Gtp** ranking as the top three predictors.

## Technologies & Libraries

-   **Language:** R
-   **Core Libraries:**
    -   `glmnet`: For Ridge and Lasso regularized regression.
    -   `randomForest`: For the Random Forest model.
    -   `xgboost`: For the high-performance Extreme Gradient Boosting model.
    -   `keras` & `tensorflow`: For building and training the Neural Network.
    -   `dplyr` & `readr`: For data manipulation and file I/O.
    -   `ggplot2`: For static visualizations.
    -   `pROC`: For calculating and analyzing AUC scores.

## Project Structure

```
smoker-prediction/
├── data/
│   └── train.csv           # The training dataset
│   └── test.csv            # The testing dataset
│   └── results_sub.csv     # The submitted dataset
├── plots/                  # Different saved plots
├── scripts/
│   └── main.R              # Main R script for the complete workflow
└── README.md                   # This README file
```

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd smoker-prediction
    ```
2.  **Install R and RStudio:**
    -   Ensure you have a recent version of R and RStudio installed.
3.  **Install R Packages:**
    -   Open the `analysis.R` script in the `R/` directory. It will contain the necessary `install.packages()` commands for all required libraries. Run these commands in your R console.

## How to Run the Analysis

1.  Open the `analysis.R` script in RStudio.
2.  Set your working directory to the root of the project folder.
3.  Run the script from top to bottom.

The script will handle all steps automatically: data loading, preprocessing, model training, evaluation, and generating the key results discussed in the report.

## Modeling Approach

A multi-model strategy was employed to identify the best classifier:
1.  **Baseline Model:** A **Logistic Regression** was first established to create a performance benchmark (AUC: 0.847).
2.  **Regularization:** **Ridge Regression** was used to address multicollinearity observed in the data (AUC: 0.828).
3.  **Ensemble Methods:** A **Random Forest** (AUC: 0.870) and an **XGBoost** model (AUC: 0.881) were implemented to capture complex, non-linear interactions between features.
4.  **Deep Learning:** A **Neural Network** with two hidden layers was built to explore a deep learning solution (AUC: 0.857).

## Results & Key Findings

-   The **XGBoost model** was the clear winner, achieving the highest predictive accuracy with an **AUC of 0.881**.
-   **Variable importance analysis** from the best models consistently showed that **hemoglobin, height, and Gtp (gamma-glutamyl transferase)** are the most significant predictors of smoking status.
-   The performance of the XGBoost model was validated, achieving a Kaggle competition score of **0.87131**.

## Future Work

-   **Advanced Neural Networks:** Explore more complex neural network architectures with additional layers or different activation functions to potentially improve performance further.
-   **Hyperparameter Tuning:** Implement more exhaustive hyperparameter tuning grids for the XGBoost and Neural Network models using techniques like Bayesian optimization.
-   **Feature Engineering:** Create new interactive features from the existing bio-signals to potentially uncover deeper patterns.

## Contact

For any questions or collaboration opportunities, please reach out:

-   **Name:** Souvik Bag
-   **Email:** `sbk29@umsystem.edu`
-   **LinkedIn:** [Your LinkedIn Profile URL]
-   **GitHub:** [https://github.com/souvik-bag](https://github.com/souvik-bag)


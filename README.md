# Data Analysis for CSGO Match Prediction

## Overview
This repository contains a comprehensive data analysis pipeline for predicting outcomes in CSGO matches. It includes data preparation, exploratory data analysis (EDA), and advanced analytics using machine learning models.


## Repository Structure

```
csgo-analytics/
├── data/                  
│   ├── raw/               # Original dataset files (if shareable)
│   └── processed/         # Cleaned and transformed datasets
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── src/                   
│   ├── data_preparation.py   # Data cleaning and feature engineering
│   ├── exploratory_data.py   # Exploratory data analysis and visualizations
│   ├── model.py              # Modeling, evaluation, and hyperparameter tuning
│   └── utils.py              # Utility functions (if needed)
├── reports/               # Visual reports, graphs, and analysis outputs
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```


## Installation & Setup
### Prerequisites
Ensure you have Python 3.8+ installed. Install required dependencies using:
```bash
pip install -r requirements.txt
```

## Data Preparation & Cleaning
- Loads dataset and checks dimensions & types
- Handles missing values, duplicates, and standardizes formats
- Performs normalization and feature scaling where necessary

Run:
```bash
python Data_Preparation_and_Cleaning.py
```

## Exploratory Data Analysis (EDA)
- Generates descriptive statistics & correlation heatmaps
- Produces visualizations (histograms, box plots, scatter plots)
- Identifies patterns and relationships in data

Run:
```bash
python Exploratory_Data_Analysis\ (EDA).py
```

## Model Training & Evaluation
- Implements regression, classification, clustering, and time series models
- Splits data into training & testing sets
- Conducts hyperparameter tuning & cross-validation

Run:
```bash
python model.py
```

## Advanced Analytics
- Builds predictive models with optimized performance
- Evaluates models using accuracy, precision, recall, and F1-score
- Produces insights and deployable solutions

Run:
```bash
python Advanced_Analytics\(buiding\ model).py
```

## Demo
To test predictions on a sample input:
```bash
python demo.py
```

## Dependencies
See `requirements.txt` for a full list of required packages, including:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Results and Analysis

- **Model Performance:** Detailed performance metrics including accuracy, F1-score, and confusion matrices are provided for each model.
- **Feature Importance:** Analysis of key features influencing match outcomes.
- **Visual Insights:** Graphical representations of data distribution, correlations, and model results.

All reports and visualizations are stored in the `reports/` directory.

---

## Contributors
- **[Cuong_Le]** – Lead Developer

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments
- Inspired by professional data science workflows
- Special thanks to open-source contributors for data analysis libraries


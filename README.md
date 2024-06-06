# Machine Learning Predictive Maintenance Project

## Overview
This project applies machine learning techniques learned in class to a synthetic predictive maintenance dataset. The aim is to predict machine failures based on various features.

## Dataset
The dataset consists of 10,000 data points with 14 features:
- **UID**: Unique identifier
- **productID**: Quality variant (L, M, H) and serial number
- **air temperature [K]**: Random walk normalized to σ=2K around 300K
- **process temperature [K]**: Random walk normalized to σ=1K added to air temperature +10K
- **rotational speed [rpm]**: Calculated from power of 2860W with noise
- **torque [Nm]**: Normally distributed around 40 Nm with σ=10 Nm, no negative values
- **tool wear [min]**: Quality variants add 5/3/2 minutes of wear

### Targets
- **Failure**: Indicates if there was a machine failure
- **Failure Type**: Specific type of failure

## Project Goals
1. **Data Preprocessing**: Clean and prepare the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**: Understand the dataset and identify important features.
3. **Feature Engineering**: Create new features or modify existing ones to improve model performance.
4. **Modeling**: Build and evaluate machine learning models to predict machine failure.
5. **Evaluation**: Assess model performance using appropriate metrics.

## Techniques Used

### Data Preprocessing
- **Loading the dataset**: Using pandas to load and explore the dataset.
- **Handling missing values**: Checking for and imputing or dropping missing values.
- **Data normalization/scaling**: Standardizing features to have a mean of 0 and standard deviation of 1 using StandardScaler from scikit-learn.
- **Encoding categorical variables**: Converting categorical features to numerical values using one-hot encoding.

### Exploratory Data Analysis (EDA)
- **Visualizations**: Using matplotlib and seaborn for plotting distributions, correlations, and feature relationships.
- **Statistical summaries**: Generating descriptive statistics to understand data distribution.

### Feature Engineering
- **Creating new features**: Deriving new features from existing ones to capture more information.
- **Feature selection**: Identifying important features using correlation matrices or feature importance from models like RandomForest.

### Model Building
- **Splitting the dataset**: Dividing the data into training and testing sets.
- **Model selection**: Trying different models such as Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and Support Vector Machines (SVM).
- **Hyperparameter tuning**: Using techniques like GridSearchCV to find the best parameters for models.

### Model Evaluation
- **Cross-validation**: Evaluating model performance using k-fold cross-validation to ensure robustness.
- **Performance metrics**: Using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to assess model performance.

### Model Interpretation
- **Feature importance**: Interpreting which features are most influential in the predictions.
- **Confusion matrix**: Analyzing the confusion matrix to understand the types of errors the model makes.

### Deployment
- **Saving the model**: Exporting the trained model using joblib or pickle for deployment.

## Files
- **final.ipynb**: Main Jupyter notebook containing the project code, including data preprocessing, EDA, modeling, and evaluation.
- **requirements.txt**: List of dependencies needed to run the project.

## Dependencies
Install the required libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt

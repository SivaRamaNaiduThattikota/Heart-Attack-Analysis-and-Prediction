# Heart Attack Analysis and Prediction

## Overview
This project aims to predict heart attack risks using various machine learning algorithms. The analysis includes data preprocessing, exploratory data analysis (EDA), and model evaluation.

## Project Structure
1. **Introduction**
   - Definition and causes of heart attacks.
   - Symptoms of heart attacks.
   - Dataset variables: age, sex, exercise-induced angina, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, and target variable indicating heart attack risk.

2. **First Organization**
   - Required Python libraries: numpy, pandas, matplotlib, seaborn.
   - Loading the dataset and initial analysis showing 303 rows and 14 columns with no missing values.

3. **Preparing for Exploratory Data Analysis (EDA)**
   - Examining missing values and unique values.
   - Separating variables into numeric and categorical.
   - Examining statistics of variables using distplot for numeric variables.

4. **Exploratory Data Analysis (EDA)**
   - Univariate analysis for numeric variables using distplot.
   - Analysis outputs indicating distributions and potential outliers.
   - Categorical variables analysis using pie charts.
   - Examining missing data and filling missing values in the 'thal' variable.
   - Bivariate analysis using FaceGrid for numeric variables and target variable.
   - Count plot for categorical variables and target variable.
   - Pair plot for examining relationships among numeric variables.
   - Feature scaling using RobustScaler.
   - Creating a new DataFrame with the melt() function.
   - Swarm plot and box plot for numerical and categorical variables.
   - Heatmap for relationships between variables.

5. **Preparation for Modeling**
   - Dropping columns with low correlation.
   - Visualizing and dealing with outliers using z-score and winsorize methods.
   - Determining distributions of numeric variables and applying transformation operations on unsymmetrical data.
   - Applying one-hot encoding to categorical variables.
   - Feature scaling with RobustScaler for machine learning algorithms.
   - Separating data into test and training sets.

6. **Modeling**
   - **Logistic Regression**
     - Used for binary classification to predict the probability of a heart attack.
     - Includes cross-validation, ROC Curve, and AUC analysis.
     - Hyperparameter optimization using GridSearchCV.
   - **Decision Tree**
     - A tree-like model used for classification.
     - Evaluated using cross-validation and ROC Curve.
   - **Support Vector Machine (SVM)**
     - A classification model that finds the optimal hyperplane to separate classes.
     - Evaluated using cross-validation and ROC Curve.
   - **Random Forest**
     - An ensemble learning method that uses multiple decision trees.
     - Includes cross-validation and ROC Curve.
     - Hyperparameter optimization using GridSearchCV.

## Installation
To run this project, you need to have Python installed along with the following libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn

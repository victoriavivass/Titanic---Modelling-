# Titanic---Modelling-
This repository is the second part of a data analysis and machine learning project focused on the Titanic dataset. 

# Titanic Data Cleaning and Random Forest Modeling
This repository contains an R script, `titanic2clean.R`, which performs data cleaning and preprocessing on the Titanic dataset, followed by implementing a **Random Forest** model to predict passenger survival. This analysis provides insights into the features that might influence survival rates and demonstrates the use of a machine learning model in R.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset Summary](#dataset-summary)
3. [Script Structure](#script-structure)
4. [Installation and Usage](#installation-and-usage)
5. [Requirements](#requirements)
6. [Results](#results)
7. [Contributions](#contributions)

## Project Description
The `titanic2clean.R` script is designed to conduct a comprehensive data preprocessing routine on the Titanic dataset and then train a Random Forest model for predicting survival. This project aims to explore the relationships between various passenger features (like age, class, and cabin presence) and survival likelihood.

## Dataset Summary
The Titanic dataset used in this project includes the following key variables:

- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Binary variable indicating survival (1 = Yes, 0 = No).
- **Pclass**: Passenger class (1st, 2nd, or 3rd).
- **Name**: Passenger's full name.
- **Sex**: Gender of the passenger.
- **Age**: Age in years.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Fare paid by the passenger.
- **Cabin**: Cabin information, which is processed into a binary variable.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Script Structure
The `titanic2clean.R` script consists of the following sections:

1. **Setup and Library Loading**:
   - Sets the working directory and clears the environment.
   - Loads necessary libraries, including `ggplot2` for visualization, `caTools` for data splitting, `caret` for model tuning, and `randomForest` for building the Random Forest model.

2. **Data Loading and Initial Exploration**:
   - Loads the Titanic dataset (`titanic_train.Rdata`).
   - Provides a summary of the dataset and performs an initial inspection of variables.

3. **Data Preprocessing**:
   - Converts `Cabin` into a binary variable (`hasCabin`) to indicate whether a passenger had a cabin listed.
   - Handles missing values and prepares other variables as needed for modeling.
   - Transforms `Survived` into a binary factor for compatibility with the Random Forest model.

4. **Random Forest Modeling**:
   - Uses the `randomForest` library to create a Random Forest model for survival prediction.
   - Splits the dataset into training and testing subsets, trains the model, and makes predictions.
   - Evaluates the model's accuracy and other performance metrics.

5. **Results and Evaluation**:
   - Outputs model evaluation metrics and generates visualizations that help interpret the model's performance.
   - Explores feature importance to understand which variables have the most influence on survival predictions.

## Installation and Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/titanic2clean.git
   cd titanic2clean

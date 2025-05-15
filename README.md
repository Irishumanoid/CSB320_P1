# CSB320 Project 1
Analyzes diabetes data from two datasets and trains multiple machine learning models for class prediction, including:

- K-Nearest Neighbors (KNN)
- Naive Bayes
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)

## Environment Setup
To install dependencies for the project by running the requirements.yml file, run the following commands in your terminal
```
conda env create -f requirements.yml
conda activate ml-env
```
To reduce code redundancy, since the same functions are run in different Juypter Notebooks corresponding to different datasets, lib.py contains utility functions for data visualization, preprocessing, and model evaluation and comparison.

## Performance Comparison
Overall, the models performed better on the secondary dataset (especially in accuracy and recall), which had more features but fewer datapoints. This is probably because more features enable the models to detect more patterns, especially in the minority class. On the secondary dataset, without SMOTE, Logistic Regression had the best performance with high accuracy (91.4%) and perfect precision (1.0), with moderate recall (46.2%). With SMOTE on the same dataset, the Decision Tree had the best overall recall (84.6%) and great accuracy (91.4%), which is ideal for detecting diabetes cases. On the provided dataset, which is smaller since it has fewer features, without SMOTE the Logistic Regression also had the most balanced performance with fairly high accuracy (72.1%) and moderate recall (53.7%). With SMOTE on the smaller dataset, KNN has the best recall (70.4%) with Logistic Regression (64.8%) coming in second. Logistic Regression and Decision Trees benefit the most from SMOTE and are the most reliable across both datasets.


The secondary dataset, which has more features, generally had higher accuracy across all the models, higher recall with SMOTE, and SMOTE was more effective overall since features were more informative and more nuanced patterns could be extrapolated. Datasets with more features are generally optimal since they allow for more patterns to be found and minority classes to be better detected.
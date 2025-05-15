import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def filter_zeros(dataset: pd.DataFrame, cols: list):
    '''
    Replace dataframe entries that contain 0 or NaN values with the median data value for that feature
    '''
    print(f'cols {cols}')
    for name in cols:
        feature_median = dataset.describe()[name]['50%']
        for i in range(len(dataset[name])):
            if dataset.loc[i, name] == 0 or pd.isna(dataset.at[i, name]):
                dataset.loc[i, name] = feature_median

def make_pairplot(df: pd.DataFrame, key: str):
    '''
    Show pairwise relationships between features and distribution estimates for individual features
    '''
    sns.pairplot(df, hue=key)

def make_corr_plot(X: pd.DataFrame, fig_x=20, fig_y=20):
    '''
    Show correlations between different features
    '''
    corr_mat = X.corr()
    plt.figure(figsize=(fig_x, fig_y))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

def make_countplot(X: pd.DataFrame, key: str):
    '''
    Show classification split for a given feature
    '''
    sns.countplot(x=key, data=X)
    plt.show()

def get_PCA_sets(X_train, X_test):
    '''
    Transform data by reducing dimensionality using Principal Component Analysis
    '''
    pca = PCA(n_components=5, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    print(pca.explained_variance_ratio_)
    return X_train, X_test

def make_feature_distribution_plots(X):
    '''
    Plot feature distributions in a grid for compactness
    '''
    cols = X.select_dtypes(include=['number']).columns
    fig, axes = plt.subplots(math.ceil(len(cols) / 3), 3, figsize=(20, 20))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax = axes[i]
        if i < len(cols):
            ax.hist(X[cols[i]], bins=20, edgecolor='black')
            ax.set_title(f'{cols[i]} Distribution')

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j]) # hide unused
        
    plt.tight_layout()
    plt.show()


def get_resampled_data(X_train, y_train):
    '''
    Get class balanced data from original training data using SMOTE
    '''
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def get_best_k_param(X_train, y_train, X_test, y_test, max_k_to_test=30):
    '''
    Get optimal k value for KNN
    '''
    best_k = 1
    best_score = 0
    for i in range(1, max_k_to_test):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, np.ravel(y_train))
        score = knn.score(X_test, np.ravel(y_test))
        print(f'k is {i} with score {score}')
        if score >= best_score:
            best_score = score
            best_k = i
    print(best_k)

def get_nb_hyperparams(X_train, y_train):
    '''
    Get optimal hyperparameters for Gaussian Native Bayes
    '''
    nb_param_grid = {
    'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
    }

    nb = GaussianNB()
    grid = GridSearchCV(estimator=nb, param_grid=nb_param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best hyperparameters:", grid.best_params_)

def get_log_reg_hyperparams(X_train, y_train):
    '''
    Get optimal hyperparameters for Logistic Regression
    '''
    log_reg_param_grid = {
        'max_iter': [50, 100, 200, 500, 1000, 2000],
        'penalty': ['l1', 'l2'],
        'C': [0.25, 0.5, 1, 1.5, 2, 2.5]
    }
    log_reg = LogisticRegression(solver='liblinear')
    grid = GridSearchCV(estimator=log_reg, param_grid=log_reg_param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best hyperparameters:", grid.best_params_)

def get_dec_tree_hyperparams(X_train, y_train):
    '''
    Get optimal hyperparameters for Decision Tree
    '''
    dec_tree_param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
    }
    dec_tree = DecisionTreeClassifier()
    grid = GridSearchCV(estimator=dec_tree, param_grid=dec_tree_param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best hyperparameters:", grid.best_params_)

def get_svm_hyperparams(X_train, y_train):
    '''
    Get optimal hyperparameters for Support Vector Classifier
    '''
    svm_param_grid = {
        'kernel': ['poly', 'rbf', 'sigmoid']
    }
    svm = SVC()
    grid = GridSearchCV(estimator=svm, param_grid=svm_param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best hyperparameters:", grid.best_params_)


def model_eval_stats(X_train, y_train, X_test, y_test, X_train_resampled, y_train_resampled, knn: KNeighborsClassifier, nb: GaussianNB, log_reg: LogisticRegression, dec_tree: DecisionTreeClassifier, svm: SVC):
    '''
    Evaluates various metrics (k-fold standard deviation and mean of f1 scores, accuracy, precision, and recall) for all models using both the original and class balanced training sets. Also plots confusion matrices representing each classifer's performance on the test set.
    '''
    for i in range(2):
        cols = ['kfold mean', 'kfold stddev', 'accuracy', 'precision', 'recall']
        cms = []
        metrics_df = pd.DataFrame(columns=cols)
        if i == 0:
            X_train_cur = X_train
            y_train_cur = y_train
        else:
            X_train_cur = X_train_resampled
            y_train_cur = y_train_resampled

        for model in [knn, nb, log_reg, dec_tree, svm]:
            model.fit(X_train_cur, np.ravel(y_train_cur))
            print(model.score(X_test, np.ravel(y_test)))
            cms.append(confusion_matrix(y_test, model.predict(X_test)))

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            results = cross_validate(model, X_train_cur, np.ravel(y_train_cur), cv=kf)

            scores = results['test_score'].round(3) 
            print(f'f1 scores are {scores}')

            mean = scores.mean().round(3)
            sd = scores.std().round(3)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            precision = precision_score(y_test, model.predict(X_test), zero_division=0)
            recall = recall_score(y_test, model.predict(X_test))

            metrics_df.loc[str(model)] = {
                'kfold mean': mean,
                'kfold stddev': sd,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        print(f'Metrics for {'original' if i == 0 else 'class balanced'} data')
        display(metrics_df)


    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, ax in enumerate(axes):
        disp = ConfusionMatrixDisplay(confusion_matrix=cms[i])
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'{cols[i]} Confusion Matrix')
    plt.tight_layout()
    plt.show()
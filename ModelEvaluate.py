# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, recall_score, f1_score, accuracy_score, precision_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

def ModelEvaluation(X_train, X_test, y_train, y_test, classifiers_tuned):
    """
    classifiers_tuned = {
    'Logistic Regression': LogisticRegression(C=0.126486),
    'SVM': SVC(C=1000, gamma=0.001),
    'KNN': KNeighborsClassifier(n_neighbors=9),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10, random_state=42),
    'Random Forest Classifier': RandomForestClassifier(max_depth=10, min_samples_leaf=2, n_estimators=200, random_state=42),
    'AdaBoost Classifier': AdaBoostClassifier(learning_rate=0.1, n_estimators=200, random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(learning_rate=0.01, min_samples_split=10, n_estimators=500, random_state=42)
    }
    """
    # Evaluation score for different ML models after Hyper parameter tuning
    model_name = []
    train_accuracy = []
    test_accuracy = []
    train_precision = []
    test_precision = []
    train_recall = []
    test_recall = []
    train_f1score = []
    test_f1score = []

    # Loop through each classifier, fit and predict
    for name, clf in classifiers_tuned.items():
        model = clf
        model.fit(X_train, y_train)
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_predict)
        test_acc = accuracy_score(y_test, y_test_predict)
        train_prec = precision_score(y_train, y_train_predict)
        test_prec = precision_score(y_test, y_test_predict)
        train_recl = recall_score(y_train, y_train_predict)
        test_recl = recall_score(y_test, y_test_predict)
        train_f1 = f1_score(y_train, y_train_predict)
        test_f1 = f1_score(y_test, y_test_predict)
        model_name.append(name)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        train_precision.append(train_prec)
        test_precision.append(test_prec)
        train_recall.append(train_recl)
        test_recall.append(test_recl)
        train_f1score.append(train_f1)
        test_f1score.append(test_f1)
        
    fig,ax = plt.subplots(figsize=(8,4))
    sns.barplot(y=model_name, x=train_accuracy, palette=['grey'], ax=ax)
    sns.barplot(y=model_name, x=test_accuracy, palette=['blue'], ax=ax)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    plt.title('Accuracy comparison of ML models')
    plt.legend(loc='lower right')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(8,4))
    sns.barplot(y=model_name, x=train_precision, palette=['grey'], ax=ax)
    sns.barplot(y=model_name, x=test_precision, palette=['blue'], ax=ax)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    plt.title('Precision comparison of ML models')
    plt.legend(loc='lower right')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(8,4))
    sns.barplot(y=model_name, x=train_recall, palette=['grey'], ax=ax)
    sns.barplot(y=model_name, x=test_recall, palette=['blue'], ax=ax)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    plt.title('Recall comparison of ML models')
    plt.legend(loc='lower right')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(8,4))
    sns.barplot(y=model_name, x=train_f1score, palette=['grey'], ax=ax)
    sns.barplot(y=model_name, x=test_f1score, palette=['blue'], ax=ax)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    plt.title('F1-score comparison of ML models')
    plt.legend(loc='lower right')
    plt.show()

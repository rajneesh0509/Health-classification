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

 
def TrainTestSplit(X, y):
    #X = df.drop(['treatment'], axis=1)
    #y = df['treatment']
    return train_test_split(X, y, test_size=0.20, random_state=43)




def ModelPreparation(X_train, y_train):
    """
    X = df.drop(['treatment'], axis=1)
    y = df['treatment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=43)
    
    """
    
    classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'AdaBoost Classifier': AdaBoostClassifier(random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)
    }
    
    model_name = []
    cross_val_accuracy = []
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        model_name.append(name)
        cross_val_accuracy.append(scores.mean())  
        
    # Comparison plot of Cross validation accuracy of different classifier models
    fig,ax = plt.subplots(figsize=(8,4))
    sns.barplot(y=model_name, x=cross_val_accuracy, palette=['grey'])
    ax.spines['right'].set_color('None')  # Removing right border
    ax.spines['top'].set_color('None')    # Removing top border
    plt.title('Cross validation accuracy comparison of ML models')
    plt.show()




def ModelTuning(X_train, y_train):
  
    classifiers = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'AdaBoost Classifier': AdaBoostClassifier(random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)
    }
        
    # defining the hyperparameters to tune for the classifiers
    params = {
    'Logistic Regression': {
        'C': np.logspace(-4, 4, 50)
    },
    'SVM': {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    },
    'KNN': {
        'n_neighbors': [3,5,7,9]
    },
    'Decision Tree': {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100],
        'criterion': ["gini", "entropy"]
    },
    'Random Forest Classifier': {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting Classifier': {
        'learning_rate': [0.01, 0.1, 0.5],
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'AdaBoost Classifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5]
    }
    }
        
    classifiers_tuned = {}
    # fit and evaluate each classifier
    for name, classifier in classifiers.items():
        print(f'\n{name}\n')
        # tune the hyperparameters using GridSearchCV
        gs = GridSearchCV(estimator=classifier, param_grid=params[name], cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        classifiers[name] = gs.best_estimator_
        classifiers_tuned[name] = gs.best_estimator_
        print('Parameter:',classifiers[name])
        #print('\nAccuracy:\n', acc_base)
        print('==================================================')
    
    return classifiers_tuned






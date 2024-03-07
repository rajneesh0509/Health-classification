import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, recall_score, f1_score, accuracy_score, precision_score, roc_auc_score, roc_curve
from datapreprocessing import DataPreprocessing
from ModelDevelop import ModelPreparation, ModelTuning
from ModelEvaluate import ModelEvaluation

# Data preprocessing and model training
df_survey = pd.read_csv("survey.csv")
df_preprocess = DataPreprocessing(df_survey)
cat_columns = df_preprocess.select_dtypes(include=['object']).columns
boolean_cols = df_preprocess.select_dtypes(include=['bool']).columns
le = LabelEncoder()
df_preprocess[cat_columns] = df_preprocess[cat_columns].apply(le.fit_transform)
df_preprocess[boolean_cols] = df_preprocess[boolean_cols].apply(le.fit_transform)
df_preprocess['age'] = MinMaxScaler().fit_transform(df_preprocess[['age']])
X = df_preprocess.drop(columns=['treatment'], axis=1)
y = df_preprocess['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=43)
#model = LogisticRegression(C=0.126486)
#model.fit(X_train, y_train)

#ModelPreparation(X_train, y_train)
#classifiers_tuned = ModelTuning(X_train, y_train)
#print(classifiers_tuned)
classifiers_tuned = {'Logistic Regression': LogisticRegression(C=1.7575106248547894), 'SVM': SVC(C=1000, gamma=0.001), 'KNN': KNeighborsClassifier(n_neighbors=7), 'Decision Tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42), 'Random Forest Classifier': RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=10,
                       random_state=42), 'AdaBoost Classifier': AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42), 'Gradient Boosting Classifier': GradientBoostingClassifier(learning_rate=0.01, min_samples_leaf=4, n_estimators=200, random_state=42)}

ModelEvaluation(X_train, X_test, y_train, y_test, classifiers_tuned)

# Importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.model_selection import train_test_split


def DataPreprocessing(df):
    # Dropping columns
    df.drop(columns=['timestamp', 'comments', 'state', 'country'], inplace=True)
    # Checking missing values after columns drop
    df_missing_check = pd.concat([df.isnull().sum(), df.nunique(), df.dtypes], axis=1).rename(columns = {0:'missingValues', 1:'uniqueCount', 2:'dataType'})
    df_missing_check.sort_values(by=['missingValues'], ascending=False)
    #Imputing missing values with Mode
    df['self_employed'] = df['self_employed'].fillna(df['self_employed'].mode()[0])
    df['work_interfere'] = df['work_interfere'].fillna(df['work_interfere'].mode()[0])
    # Unique values of Age
    df['age'].unique()
    # Dropping entries corresponding to negative age or age > 100 years
    df.drop(df[df['age'] < 0].index, inplace=True)
    df.drop(df[df['age'] > 100].index, inplace=True)
    # Removing outlier (age < 15years) in Tech workplace
    df.drop(df[df['age'] < 15].index, inplace=True)
    # Checking Unique values of Gender
    df['gender'].unique()
    # Arranging "Gender" in proper format
    df['gender'] = df['gender'].replace(['M', 'Male', 'm', 'Male-ish', 'maile', 'Mal', 'Male (CIS)', 'Make', 'male leaning androgynous', 'Male ', 'Man', 'msle', 'Mail', 'cis male', 'Malr', 'Cis Man', 'male',], 'Male')
    df['gender'] = df['gender'].replace(['Female', 'female', 'Cis Female', 'F', 'Woman', 'f','Femake', 'woman', 'Female ', 'cis-female/femme', 'Female (cis)', 'femail'], 'Female')
    df['gender'] = df['gender'].replace(['Trans-female', 'something kinda male?', 'Cis Male', 'queer/she/they', 'non-binary', 'Nah', 'Enby', 'fluid', 'Genderqueer', 'Androgyne', 'Agender', 'Guy (-ish) ^_^', 'Trans woman','Neuter', 'Female (trans)', 'queer', 'A little about you','ostensibly male, unsure what that really means'], 'Other')
    # Checking duplicate entries
    df[df.duplicated()==True]
    # Dropping duplicate entries
    df.drop_duplicates(inplace=True)
    return df


def Encoding(df):
    categorical_cols = ['gender', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'no_employees',
                        'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
                        'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                        'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']
    label_df = df.copy()
    df_in = df[categorical_cols].copy()
    encoded_dict = dict()
    le = LabelEncoder()
    for cat in categorical_cols:
        #df_in[cat] = df_in[cat].str.lstrip()
        le = le.fit(list(df_in[cat]) + ['Unknown'])
        encoded_dict[cat] = [cat for cat in le.classes_]
        label_df[cat] = le.transform(df_in[cat])
    return label_df, encoded_dict





    

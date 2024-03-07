import streamlit as st
import pandas as pd
import datetime
from base64 import b64encode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from datapreprocessing import DataPreprocessing
from batch_processing_data import generate_data

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
model = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=10, random_state=42)
#model = AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Title of streamlit web application
st.title('Tech workplace Health classification Application')

# Setting background image for web application
def get_base64_of_binfile(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return b64encode(data).decode()

def set_bg_page(png_file):
    bin_str = get_base64_of_binfile(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_bg_page("workplace survey.png")
    
 

choice = st.sidebar.radio("Choose the option", options=("Manual entry", "File upload", "Batch streaming"))
if choice == "File upload":
    # User input as a file upload from web application
    uploaded_file = st.file_uploader("Please upload a csv file")
    # Prediction on user input data
    if st.button("Predict"):
        if uploaded_file is None:
            st.write("You have not uploaded any file. Please upload a csv file.")
        else:
            df_test = pd.read_csv(uploaded_file)
            df_test_copy = df_test.copy()
            df_test_preprocess = DataPreprocessing(df_test)
            cat_columns_test = df_test_preprocess.select_dtypes(include=['object']).columns
            boolean_cols_test = df_test_preprocess.select_dtypes(include=['bool']).columns
            df_test_preprocess[cat_columns_test] = df_test_preprocess[cat_columns_test].apply(le.fit_transform)
            df_test_preprocess[boolean_cols_test] = df_test_preprocess[boolean_cols_test].apply(le.fit_transform)
            prediction = model.predict(df_test_preprocess)
            predict_series = pd.Series(prediction)
            predict_series = predict_series.map({0: 'No', 1: 'Yes'})
            predict_df = pd.DataFrame(predict_series, columns=['treatment'])
            df_test_with_output = pd.concat([df_test_copy, predict_df], axis=1, sort=False)
            st.write("Result/Recommendation: ", df_test_with_output)

elif choice == "Manual entry":
    # User input directly on GUI home page
    timestamp = st.date_input("DateTime", datetime.datetime.now())
    age = st.number_input("Age", min_value=0, max_value=100, value=18, step=1, placeholder="Enter your age...")
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ['Canada','United States','United Kingdom','Bulgaria','France','Portugal',
                        'Netherlands','Switzerland','Poland','Australia','Germany','Russia','Mexico','Brazil',
                        'Slovenia','Costa Rica','Austria','Ireland','India','South Africa','Italy','Sweden',
                        'Colombia','Latvia','Romania','Belgium','New Zealand','Zimbabwe','Spain','Finland',
                        'Uruguay','Israel','Bosnia and Herzegovina','Hungary','Singapore','Japan','Nigeria',
                        'Croatia','Norway','Thailand','Denmark','Bahamas, The','Greece','Moldova','Georgia',
                        'China','Czech Republic','Philippines'])
    state = st.selectbox("State", ['IN','IL','NA','TX','TN','MI','OH','CA','CT','MD','NY','NC','MA','IA','PA',
                        'WA','WI','UT','NM','OR','FL','MN','MO','AZ','CO','GA','DC','NE','WV','OK','KS','VA',
                        'NH','KY','AL','NV','NJ','SC','VT','SD','ID','MS','RI','WY','LA','ME'])
    self_employed = st.selectbox("Self employed", ["Yes", "No"])
    family_history = st.selectbox("Family history of mental sickness", ["TRUE", "FALSE"])
    work_interfere = st.selectbox("Work interference", ["Never", "Often", "Rarely", "Sometimes"])
    no_employees = st.selectbox("No. of employees", ['1-5', '6-25' '26-100', '100-500', '500-1000', 'More than 1000'])
    remote_work = st.selectbox("Work remotely", ["TRUE", "FALSE"])
    tech_company = st.selectbox("Tech company organization", ["TRUE", "FALSE"])
    benefits = st.selectbox("Benefits from employer", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Mental health care from employer", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Employee welness program", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Seek help", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Anonymity protected", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of medical leave", ["Somewhat easy", "Somewhat difficult", "Very easy", "Very difficult", "Don't know"])
    mental_health_consequence = st.selectbox("Mental health consequence", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Physical health consequence", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Discuss mental health issue with coworkers", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Discuss mental health issue with supervisor", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Mental health interview", ["Yes", "No", "Maybe"])
    phys_health_interview = st.selectbox("Physical health interview", ["Yes", "No", "Maybe"])
    mental_vs_physical = st.selectbox("Employer takes mental health as serious as physical health", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Heard any consequences in workplace", ["TRUE", "FALSE"])
    comments = st.text_input("Comment if any", max_chars=100, placeholder="Enter comment here...")

    if st.button("Predict"):
        data = {'timestamp': timestamp, 'age': age, 'gender': gender, 'country': country, 'state': state, 'self_employed': self_employed, 'family_history':family_history,
            'work_interfere': work_interfere, 'no_employees': no_employees, 'remote_work':remote_work, 'tech_company': tech_company, 'benefits': benefits, 'care_options': care_options,
            'wellness_program': wellness_program, 'seek_help': seek_help, 'anonymity': anonymity, 'leave': leave, 'mental_health_consequence': mental_health_consequence,
            'phys_health_consequence': phys_health_consequence, 'coworkers': coworkers, 'supervisor': supervisor, 'mental_health_interview': mental_health_interview,
            'phys_health_interview': phys_health_interview, 'mental_vs_physical': mental_vs_physical, 'obs_consequence': obs_consequence, 'comments': comments}
        df_manual = pd.DataFrame([list(data.values())], columns = ['timestamp', 'age', 'gender', 'country', 'state', 'self_employed', 'family_history', 'work_interfere', 'no_employees',
                'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
                'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence', 'comments'])
        df_manual_copy = df_manual.copy()
        df_manual_preprocess = DataPreprocessing(df_manual)
        cat_columns_manual = df_manual_preprocess.select_dtypes(include=['object']).columns
        boolean_cols_manual = df_manual_preprocess.select_dtypes(include=['bool']).columns
        df_manual_preprocess[cat_columns_manual] = df_manual_preprocess[cat_columns_manual].apply(le.fit_transform)
        df_manual_preprocess[boolean_cols_manual] = df_manual_preprocess[boolean_cols_manual].apply(le.fit_transform)
        prediction_manual = model.predict(df_manual_preprocess)
        if prediction_manual == 0:
            st.write("Your health is fine.")
        else:
            st.write("You are recommended to go for Mental health checkup.")

elif choice == "Batch streaming":
    data_point_count = st.number_input("Please enter batch size of real-time data.", min_value=1, placeholder="Enter a number here")
    df_batch = generate_data(data_point_count)
    if st.button("Predict"):
        df_batch_copy = df_batch.copy()
        df_batch_preprocess = DataPreprocessing(df_batch)
        cat_columns_batch = df_batch_preprocess.select_dtypes(include=['object']).columns
        boolean_cols_batch = df_batch_preprocess.select_dtypes(include=['bool']).columns
        df_batch_preprocess[cat_columns_batch] = df_batch_preprocess[cat_columns_batch].apply(le.fit_transform)
        df_batch_preprocess[boolean_cols_batch] = df_batch_preprocess[boolean_cols_batch].apply(le.fit_transform)
        prediction_batch = model.predict(df_batch_preprocess)
        predict_batch_series = pd.Series(prediction_batch)
        predict_batch_series = predict_batch_series.map({0: 'No', 1: 'Yes'})
        predict_batch_df = pd.DataFrame(predict_batch_series, columns=['treatment'])
        df_batch_with_output = pd.concat([df_batch_copy, predict_batch_df], axis=1, sort=False)
        st.write("Result/Recommendation: ", df_batch_with_output)

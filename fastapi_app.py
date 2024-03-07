import pandas as pd
import uvicorn
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import StringIO
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from datapreprocessing import DataPreprocessing
#from batch_processing_data import generate_data

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

# Create FastAPI instance
app = FastAPI()

class model_input(BaseModel):
    timestamp: datetime
    age: int
    gender: str
    country: str
    state: str
    self_employed: str
    family_history: str
    work_interfere: str
    no_employees: str
    remote_work: str
    tech_company: str
    benefits: str
    care_options: str
    wellness_program: str
    seek_help: str
    anonymity: str
    leave: str
    mental_health_consequence: str
    phys_health_consequence: str
    coworkers: str
    supervisor: str
    mental_health_interview: str
    phys_health_interview: str
    mental_vs_physical: str
    obs_consequence: str
    comments: str

class Input(BaseModel):
    data: List[model_input]
    def return_dict_input(cls, ):
        return [inp.model_dump() for inp in cls.data]
    
def process_csv_json(contents, file_type, valid_formats):
    # Read the file contents as a byte string
    contents = contents.decode()  # Decode the byte string to a regular string
    new_columns = ['timestamp', 'age', 'gender', 'country', 'state', 'self_employed', 'family_history', 'work_interfere', 'no_employees',
                'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
                'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence', 'comments']
    # Process the uploaded file
    if file_type == valid_formats[0]:
        data = pd.read_csv(StringIO(contents))
    elif file_type == valid_formats[1]:
        data = pd.read_json(contents)
    dict_new_old_cols = dict(zip(data.columns, new_columns)) # get dict of new and old cols
    data = data.rename(columns = dict_new_old_cols)
    return data


@app.get("/")
def home():
    return {'msg': 'Welcome to Tech Workplace health survey classification'}

@app.post("/predict")
async def predict(timestamp: datetime, age: int, gender: str, country: str, state: str, self_employed: str, family_history: str, work_interfere: str, no_employees: str,
                remote_work: str, tech_company: str, benefits: str, care_options: str, wellness_program: str, seek_help: str, anonymity: str, leave: str,
                mental_health_consequence: str, phys_health_consequence: str, coworkers: str, supervisor: str, mental_health_interview: str, phys_health_interview: str,
                mental_vs_physical: str, obs_consequence: str, comments: str):
    Columns = ['timestamp', 'age', 'gender', 'country', 'state', 'self_employed', 'family_history', 'work_interfere', 'no_employees',
                'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
                'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence', 'comments']
    data = pd.DataFrame([[timestamp, age, gender, country, state, self_employed, family_history, work_interfere, no_employees,
                        remote_work, tech_company, benefits, care_options, wellness_program, seek_help, anonymity, leave,
                        mental_health_consequence, phys_health_consequence, coworkers, supervisor, mental_health_interview, phys_health_interview,
                        mental_vs_physical, obs_consequence, comments]], columns=Columns)
    df_test_preprocess = DataPreprocessing(data)
    cat_columns_test = df_test_preprocess.select_dtypes(include=['object']).columns
    boolean_cols_test = df_test_preprocess.select_dtypes(include=['bool']).columns
    df_test_preprocess[cat_columns_test] = df_test_preprocess[cat_columns_test].apply(le.fit_transform)
    df_test_preprocess[boolean_cols_test] = df_test_preprocess[boolean_cols_test].apply(le.fit_transform)
    prediction_manual = model.predict(df_test_preprocess)
    if prediction_manual == 0:
        return "Your health is fine."
    else:
        return "You are recommended to go for Mental health checkup."


@app.post("/predict-manual")
async def predict_manual(inputs: Input):
    data = pd.DataFrame(inputs.return_dict_input())
    df_manual_preprocess = DataPreprocessing(data)
    cat_columns_manual = df_manual_preprocess.select_dtypes(include=['object']).columns
    boolean_cols_manual = df_manual_preprocess.select_dtypes(include=['bool']).columns
    df_manual_preprocess[cat_columns_manual] = df_manual_preprocess[cat_columns_manual].apply(le.fit_transform)
    df_manual_preprocess[boolean_cols_manual] = df_manual_preprocess[boolean_cols_manual].apply(le.fit_transform)
    prediction_manual = model.predict(df_manual_preprocess)
    if prediction_manual == 0:
        return "Your health is fine."
    else:
        return "You are recommended to go for Mental health checkup."


@app.post("/predict-upload")
async def predict_upload(file: UploadFile=File(...)):
    file_type = file.content_type # Type of the uploaded file
    valid_formats = ['text/csv', 'application/json'] # List contains valid formats that API can accept
    if file_type not in valid_formats:
        return JSONResponse(content={"error": f"File is not in valid format. It must be one of: {', '.join(valid_formats)}"})
    else:
        contents = await file.read() # Check contents of file
        data= process_csv_json(contents=contents, file_type=file_type, valid_formats=valid_formats)
        #data = pd.read_csv(file.file)
        df_upload_preprocess = DataPreprocessing(data)
        cat_columns_upload = df_upload_preprocess.select_dtypes(include=['object']).columns
        boolean_cols_upload = df_upload_preprocess.select_dtypes(include=['bool']).columns
        df_upload_preprocess[cat_columns_upload] = df_upload_preprocess[cat_columns_upload].apply(le.fit_transform)
        df_upload_preprocess[boolean_cols_upload] = df_upload_preprocess[boolean_cols_upload].apply(le.fit_transform)
        prediction_upload = model.predict(df_upload_preprocess)
        return JSONResponse(content={"Prediction": prediction_upload.tolist()})
        


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


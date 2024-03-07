import pandas as pd
import time
from datetime import datetime
import random
import string

# Function to generate random near-real time (batch) data
def generate_data(BatchSize):
    data = []
    for i in range(0, BatchSize):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        age = random.randint(15, 70)
        gender = random.choice(["Male", "Female"])
        country = random.choice(["United States", "United Kingdom", "Canada", "India", "Russia", "Germany", "Netherlands", "Ireland", "Australia"])
        state = random.choice(["NA", "CA", "WA", "NY", "TN", "TX", "OH", "IL", "OR", "PA", "IN", "MI", "MN", "MA"])
        self_employed = random.choice(["Yes", "No"])
        family_history = random.choice(["TRUE", "FALSE"])
        work_interfere = random.choice(["NA", "Often", "Rarely", "Never", "Sometimes"])
        no_employees = random.choice(["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
        remote_work = random.choice(["TRUE", "FALSE"])
        tech_company = random.choice(["TRUE", "FALSE"])
        benefits = random.choice(["Yes", "No", "Don't know"])
        care_options = random.choice(["Yes", "No", "Not sure"])
        wellness_program = random.choice(["Yes", "No", "Don't know"])
        seek_help = random.choice(["Yes", "No", "Don't know"])
        anonymity = random.choice(["Yes", "No", "Don't know"])
        leave = random.choice(["Somewhat easy", "Somewhat difficult", "Very easy", "Very difficult", "Don't know"])
        mental_health_consequence = random.choice(["Yes", "No", "Maybe"])
        phys_health_consequence = random.choice(["Yes", "No", "Maybe"])
        coworkers = random.choice(["Yes", "No", "Some of them"])
        supervisor = random.choice(["Yes", "No", "Some of them"])
        mental_health_interview = random.choice(["Yes", "No", "Maybe"])
        phys_health_interview = random.choice(["Yes", "No", "Maybe"])
        mental_vs_physical = random.choice(["Yes", "No", "Don't know"])
        obs_consequence = random.choice(["TRUE", "FALSE"])
        random_str = ''.join(random.choices(string.ascii_letters, k=10))
        comments = str(random_str)
        
        data.append((timestamp, age, gender, country, state, self_employed, family_history, work_interfere, no_employees, 
                    remote_work, tech_company, benefits, care_options, wellness_program, seek_help, anonymity, leave, 
                    mental_health_consequence, phys_health_consequence, coworkers, supervisor, mental_health_interview, 
                    phys_health_interview, mental_vs_physical, obs_consequence, comments))
        i += 1
        time.sleep(random.uniform(1, 5))
    df_batch = pd.DataFrame(data, columns = ['timestamp', 'age', 'gender', 'country', 'state', 'self_employed', 'family_history', 'work_interfere', 'no_employees', 
                    'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 
                    'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 
                    'phys_health_interview', 'mental_vs_physical', 'obs_consequence', 'comments'])
    return df_batch

df_batch = generate_data(10)
df_batch.to_csv('generated_dataset.csv', index = False)
#print(df_batch)


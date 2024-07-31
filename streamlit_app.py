import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
import google.generativeai as genai

genai.configure(api_key='AIzaSyBs5rT5G2cM-d2p_Un15THLq1Q7tYsJ9kU')
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='WeLift dashboard',
    layout = 'wide',
    page_icon= "🦾"
)
st.Title('Engagement Analysis')
# Load the model
model = joblib.load('RF_Engagement.joblib')

# Load data (optional)
data = pd.read_csv('combine.csv')

input_col = ['Title', 'IsContract', 'IsTemporary', 'IsPartTime', 'IsFullTime',
       'TerminationType', 'DepartmentType', 'Division', 'State',
       'JobFunctionDescription', 'GenderCode', 'RaceDesc', 'MaritalDesc',
       'Work-Life Balance Score', 'Training Program Name',
       'Training Type', 'Training Duration(Days)', 'Training Cost', 'Age',
       'TrainingComplete']
input_data = {key: 0 for key in input_col}
TitleOptions = ['Production Technician I', 'Area Sales Manager',
       'Production Technician II', 'IT Support', 'Network Engineer',
       'Sr. Network Engineer', 'Principal Data Architect',
       'Enterprise Architect', 'Sr. DBA', 'Database Administrator',
       'Data Analyst', 'Data Analyst ', 'Data Architect', 'CIO',
       'BI Director', 'Sr. Accountant', 'Software Engineering Manager',
       'Software Engineer', 'Shared Services Manager',
       'Senior BI Developer', 'Production Manager', 'President & CEO',
       'Administrative Assistant', 'Accountant I', 'BI Developer',
       'Sales Manager', 'IT Manager - Support', 'IT Manager - Infra',
       'IT Manager - DB', 'Director of Sales', 'Director of Operations',
       'IT Director']
TerminationTypeOptions = ['Unk', 'Involuntary', 'Resignation', 'Retirement', 'Voluntary']
DepartmentTypeOptions = ['Production       ', 'Sales', 'IT/IS', 'Executive Office','Software Engineering', 'Admin Offices']
DivisionOptions = ['Finance & Accounting', 'Aerial', 'General - Sga', 'General - Con',
       'Field Operations', 'General - Eng', 'Engineers', 'Executive',
       'Splicing', 'Project Management - Con', 'Fielders',
       'Project Management - Eng', 'Shop (Fleet)',
       'Wireline Construction', 'Catv', 'Yard (Material Handling)',
       'Wireless', 'People Services', 'Underground',
       'Billable Consultants', 'Technology / It', 'Sales & Marketing',
       'Safety', 'Isp', 'Corp Operations']
# Function to preprocess input data (including label encoding)
def preprocess_input(input_data):
    preprocessed_data = []
    for key, value in input_data.items():
        if type(value) == str:
            le = joblib.load(f'le_{key}.joblib')
            input_data[key] = le.transform([value])[0]
    for k in input_col:
        preprocessed_data.append(input_data[k])
    return preprocessed_data

# Main app
def main():
    
    st.write('# Analysis Results')

    prediction = None

    # Sidebar inputs
    with st.sidebar:
        Name = st.text_input('Full Name')
        StartDate = st.date_input("Enrollment Start Date", datetime.date.today())
        
        input_data['Title'] = st.selectbox('Previous Job Title', options=TitleOptions)
        
        job_type = st.selectbox("Was previous job a contract?", ("Yes", "No"))
        input_data['IsContract'] = 1 if job_type == "Yes" else 0
    
        job_type = st.selectbox("Type of previous job",("Temporary", "PartTime", "FullTime"))
        input_data[f'Is{job_type}'] = 1
                                
        input_data['TerminationType'] = st.selectbox('Type of Previous Job Termination',options= TerminationTypeOptions)

        input_data['DepartmentType'] = st.selectbox('Previous Job Department',options=DepartmentTypeOptions)

        input_data['Division'] = st.selectbox('Previous Job Division', options=DivisionOptions)



    # Make prediction
    prediction = model.predict([preprocess_input(input_data)])[0]

    if prediction != None:
        # Visualize prediction on box plot
        fig, ax = plt.subplots()
        ax.boxplot(data['Engagement Score'])  # Replace with your data
        ax.scatter(1, prediction, color='red')
        ax.text(1, prediction, f'{prediction:.2f}')
        st.pyplot(fig)

    promptScript = [f'according to the input data {input_data} and the predicted engagement score {prediction} over 5, write a short report of the clint']
      
    gen_model = genai.GenerativeModel('gemini-pro')
                  
    with st.spinner("Generating..."):
        response = gen_model.generate_content(promptScript,request_options={"timeout": 600})
        
    with st.container():      
        st.write(response.text)
      
    @st.cache_data
    def to_text():
        return "### Script ###\n\n" + responseScript.text

    btn = st.download_button(
        label="Download Report",
        data=to_text(),
        file_name=f"{Name}_Report.txt"
    )

if __name__ == '__main__':
    main()
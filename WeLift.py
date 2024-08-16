import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import google.generativeai as genai

genai.configure(api_key='AIzaSyBs5rT5G2cM-d2p_Un15THLq1Q7tYsJ9kU')
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='WeLift dashboard',
    #layout = 'wide',
    page_icon= "🦾"
)

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
    
    st.write('# Analysis Results Generated by ML and Gemini')

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

        new_client = pd.DataFrame()
        for c in data.columns:
            if c in input_col:
                new_client[c] = input_data[c]
            else:
                new_client[c] = data[c].mode().iloc[0]

    # Make prediction
    prediction = model.predict([preprocess_input(input_data)])[0]
    new_client['Engagement Score'][0] = prediction

    if prediction != None:
        # 创建两个容器，用于并排显示图表
        col1, col2 = st.columns(2)

        # 生成改进后的图表
        with col1:
            # 将新数据添加到原始数据中
            
            data_new = pd.concat([data, new_client], ignore_index=True)

            # 绘制改进后的图表，并标注新数据
            sns.set_theme(style="whitegrid")
            #cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
            
            g = sns.relplot(
                data=data_new,
                x="Performance Score", y="Engagement Score",
                hue="GenderCode", size="Training Cost",
                #palette=cmap, 
                sizes=(10, 200),
            )
            g.set(xscale="log", yscale="log")
            g.ax.xaxis.grid(True, "minor", linewidth=.25)
            g.ax.yaxis.grid(True, "minor", linewidth=.25)
            g.despine(left=True, bottom=True)

            # Highlight new client data point
            if len(new_client) > 0:
                new_client_x = new_client["Performance Score"].iloc[0]
                new_client_y = new_client["Engagement Score"].iloc[0]
                plt.scatter(new_client_x, new_client_y, color='red', s=100, label='New Client')

                # Add annotation with arrow pointing to new client
                plt.annotate("New Client", (new_client_x, new_client_y),
                        xytext=(new_client_x + 0.2, new_client_y + 0.2),
                        arrowprops=dict(facecolor='red', shrink=0.05))

                # Add legend to differentiate new client point
                plt.legend()
                st.pyplot(plt)
            else:
                # Handle the case where there's no data in new_client (e.g., set a default value)
                new_client_x = None  # Or any appropriate default
                new_client_y = None
                # Optionally display a message if there's no new client data
                plt.text(data['Performance Score'].mean(), data['Engagement Score'].mean(),
                         "No new client data available", ha='center', va='center')
    
            
            

        # Right column - Box plot of new client's Engagement Score
        with col2:
            sns.set_theme(style="whitegrid")
            sns.boxplot(
                x="variable",
                y="Engagement Score",
                showmeans=True,
                data=pd.DataFrame({'variable': ['All Data', 'New Client'], 
                                   'Engagement Score': [data['Engagement Score'].median(), 
                                                        new_client['Engagement Score'].iloc[0]]}
                                                        )
                )
            st.pyplot(plt)

            

    promptScript = [f'according to the new client data {new_client} and the predicted engagement score {prediction} over 5, write a short report of the client']
      
    gen_model = genai.GenerativeModel('gemini-pro')
                  
    with st.spinner("Generating..."):
        response = gen_model.generate_content(promptScript,request_options={"timeout": 600})
        
    with st.container():      
        st.write(response.text)
      
    @st.cache_data
    def to_text():
        return "### Script ###\n\n" + response.text

    btn = st.download_button(
        label="Download Report",
        data=to_text(),
        file_name=f"{Name}_Report.txt"
    )

if __name__ == '__main__':
    main()
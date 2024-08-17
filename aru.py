import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle as pk

model=pk.load(open('disease.pkl','rb'))


st.set_page_config(page_title='Heart_Disease_Prediction',layout='wide')

df=pd.read_csv('heart.csv')

c1,c2,c3=st.columns(3)
c2.title('Hear Disease Prediction ')
st.markdown('________')

a1,a2,a3=st.columns((3))
with a1:
        
    age=st.number_input('Age',28,100)
    sex=st.radio('Gender of the patient :',('Male','Female'),horizontal=True)
    cp=st.selectbox('Chest pain type',('Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'))
    trestbps=st.slider('Resting blood pressure in mm Hg',94,200)
    chol=st.slider('Serum cholesterol in mg/dl',126,564)

with a2 :
    fbs=st.radio('Fasting blood sugar level, categorized as above 120 mg/dl ',('YES','No'),horizontal=True)
    restecg=st.selectbox('Resting electrocardiographic results:',('Normal','Having ST-T wave abnormality','Showing probable or definite left ventricular hypertrophy'))
    thalach=st.slider('Maximum heart rate achieved during a stress test',71,202)
    exang=st.radio('Exercise-induced angina:', ('YES','NO'))
with a3:
    oldpeak=st.slider('ST depression induced by exercise relative to rest',0,7)
    slope=st.selectbox('Slope of the peak exercise ST segment:',('Upsloping','Flat','Downsloping'))
    ca=st.slider('Number of major vessels (0-4) colored by fluoroscopy',0,4)
    thal=st.selectbox('Thalium stress test result:',('Normal','Fixed defect','Reversible defect','Not described'))
    
input_option=pd.DataFrame(
    [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
    columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
    )
with st.expander('You selected options'):
    st.dataframe(input_option)
input_option['sex'].replace(['Male','Female'],[0,1],inplace=True)
input_option['cp'].replace(['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'],[0,1,2,3],inplace=True)
input_option['fbs'].replace(['YES','No'],[1,0],inplace=True)
input_option['restecg'].replace(['Normal','Having ST-T wave abnormality','Showing probable or definite left ventricular hypertrophy'],[0,1,2],inplace=True)
input_option['exang'].replace(['YES','NO'],[1,0],inplace=True)
input_option['slope'].replace(['Upsloping','Flat','Downsloping'],[0,1,2],inplace=True)
input_option['thal'].replace(['Normal','Fixed defect','Reversible defect','Not described'],[0,1,2,3],inplace=True)

b1,b2,b3=st.columns(3)
if b2.button('predict'):
    disease=model.predict(input_option)
    if disease==0:
        st.success('Hey you Don"t have Heart disease😊')
    else:
        st.error('Hey you have heart disease😞')


attribution="""
    - Age of the patient in years
    - Gender of the patient (0 = male, 1 = female)
    - Chest pain type:(0: Typical angina,1: Atypical angina,2: Non-anginal pain,3: Asymptomatic)
    - Resting blood pressure in mm Hg
    - Serum cholesterol in mg/dl
    - Fasting blood sugar level, categorized as above 120 mg/dl (1 = true, 0 = false)
    - Resting electrocardiographic results:(0: Normal,1: Having ST-T wave abnormality,2: Showing probable or definite left ventricular hypertrophy)
    - Maximum heart rate achieved during a stress test
    - Exercise-induced angina (1 = yes, 0 = no)
    - ST depression induced by exercise relative to rest
    - Slope of the peak exercise ST segment:(0: Upsloping,1: Flat,2: Downsloping)
    - Number of major vessels (0-4) colored by fluoroscopy
    - Thalium stress test result:(0: Normal,1: Fixed defect,2: Reversible defect,3: Not described)
    - Heart disease status (0 = no disease, 1 = presence of disease)
"""
with st.expander('Notes :'):
    st.markdown(attribution)
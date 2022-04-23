import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from PIL import Image
import streamlit as st



st.write("DIABATIES DETECTION")
df=pd.read_csv('D:\LIL\SR\ML\diabetes.csv')
st.subheader('DATA INFORMATION')
st.dataframe(df)
st.write(df.describe())
chart=st.bar_chart(df)
X=df.iloc[:,0:8].values
Y=df.iloc[:,-1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

def get_user_input():
    Pregnancies=st.sidebar.slider('Pregnancies',0,17,3)
    Glucose=st.sidebar.slider('Glucose',0,199,117)
    BloodPressure=st.sidebar.slider('BloodPressure',0,122,72)
    SkinThickness=st.sidebar.slider('SkinThickness',0,99,23)
    Insulin=st.sidebar.slider('Insulin',0,846,30)
    BMI=st.sidebar.slider('BMI',0,67,32)
    DiabetesPedigreeFunction=st.sidebar.slider('DiabetesPedigreeFunction',0.0,2.42,0.3725)
    Age=st.sidebar.slider('Age',21,81,29)

    user_dictionary={'Pregnancies':Pregnancies,
                     'Glucose':Glucose,
                     'BloodPressure':BloodPressure,
                     'SkinThickness':SkinThickness,
                     'Insulin':Insulin,
                     'BMI':BMI,
                     'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
                     'Age':Age
    }
    features=pd.DataFrame(user_dictionary,index=[0])
    return features
user_input=get_user_input()
st.subheader("USER INPUT")
RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)
st.subheader("Model Test Accuracy Score")
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+str("%"))
prediction=RandomForestClassifier.predict(user_input)
st.subheader('Classification')
st.write(prediction)

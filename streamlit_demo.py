# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 15:46:05 2025

@author: sait6
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open("trained_model.sav",'rb'))

#creating a function for prediction
def cancer_data(input):
    asarr=np.asarray(input)
    resh=asarr.reshape(1,-1)
    prediction=loaded_model.predict(resh)
    print(prediction)
    if(prediction[0]==0):
        return "does not have breast cancer"
    else:
        return "does have Breast Cancer"
def main():
    st.title("cancer prediction")
    #getting the input data from the user
   # mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,diagnosis
    mean_radius= st.number_input("enter mean_radius value")
    mean_texture= st.number_input("enter the mean_texture value")
    mean_perimeter=  st.number_input("enter the mean_perimeter value")
    mean_area= st.number_input("enter the mean_area value")
    mean_smoothness= st.number_input("enter the mean_smoothness value")
    #code for prediction
    diagnosis=''
    #creating a button for prediction
    if st.button("submit"):
        diagnosis=cancer_data([mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness])
    st.success(diagnosis)
if __name__ == '__main__': 
    main()
    

    

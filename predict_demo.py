# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 22:28:05 2024

@author: sait6
"""

import pandas as pd
import numpy as np
import pickle
loaded_model=pickle.load(open("C:/Users/sait6/Downloads\streamlit_demo/trained_model.sav",'rb'))
#predictive system
input=(20.29,14.34,135.1,1297.0,0.1003)
asarr=np.asarray(input)
resh=asarr.reshape(1,-1)
prediction=loaded_model.predict(resh)
print(prediction)
if(prediction[0]==0):
  print('does not have Breast Cancer')
else:
  print('Does have Breast Cancer')
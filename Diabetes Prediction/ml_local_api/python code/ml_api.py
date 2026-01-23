# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 10:10:04 2026

@author: sdnza
"""


from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

class model_input(BaseModel):  #fast api checks the input using this.
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction:float
    Age: int
    
#loading the model

diabetes_model, scaler = pickle.load(
    open(r"C:\Users\sdnza\OneDrive\Masaüstü\ml model api\python code\trained_model (1).sav", "rb")
)


@app.post("/diabetes_prediction")
def diabetes_predd(input_parameters:model_input):#it checks whether input meets the
#model_input structure.

#it comes as json from client but to check its structure, we convert it innto python object (when
#comparing it with base model)

    input_data=input_parameters.json() #so we need to convert to json again
    input_dictionary=json.loads(input_data) #converts json string to python objetc
    #now its python dict.
    preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    input_data_reshaped=scaler.transform(input_list)
    
    pred=diabetes_model.predict([input_data_reshaped])
    if (pred[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
    
    
    

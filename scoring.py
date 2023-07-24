from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from datetime import datetime



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    model = pickle.load(open(os.path.join(model_path,'trainedmodel.pkl'),'rb'))

    filepath = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(filepath)
    X_test = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = df['exited']
    
    # Make prediction
    y_pred = model.predict(X_test)

    # evaluate model on test set
    f1_score = metrics.f1_score(y_test, y_pred)

    # Write data with timestamp
    with open(os.path.join(model_path, 'latestscore.txt'), "w") as file: 
        file.write(f"f1 score = {f1_score}")

    return f1_score

if __name__ == '__main__':
    score_model()
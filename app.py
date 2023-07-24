from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics, scoring 
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
#app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

#######################Welcome check
@app.route('/')
def index():
    return "Welcome to the app"

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','POST','OPTIONS'])
def predict():        
    if request.method == 'POST':
        file = request.files['filename']
        dataset = pd.read_csv(file)
        return diagnostics.model_predictions(dataset)
    if request.method == 'GET':
        file = request.args.get('filename')
        dataset = pd.read_csv(file)
        return jsonify(diagnostics.model_predictions(dataset).tolist())

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    return {'f1 score': scoring.score_model()} #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats_summary():        
    #check means, medians, and modes for each column
    df_statistics = diagnostics.dataframe_summary()
    return jsonify(df_statistics) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats_diagnostics():        
    #check timing and percent NA values
    missing_data = diagnostics.perc_missing_values()
    timing = diagnostics.execution_time()
    dependency_check = diagnostics.outdated_packages_list().to_dict(orient='records')
    return {'execution time': {step:duration 
                for step, duration in zip(['ingestion time','training time'],
                                            timing)}, 
            'percentage missing values': {col:pct 
                for col, pct in zip(['lastmonth_activity',
                                    'lastyear_activity',
                                    'number_of_employees',
                                    'exited'], missing_data)},
            'dependency check':[dependency_check]
            } #add return value for all diagnostics


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)

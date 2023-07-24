import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

###################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 
filepath = os.path.join(test_data_path,'testdata.csv')

#Call each API endpoint and store the responses
response1 = requests.get(f'{URL}/prediction' + f'?filename={filepath}').text
response2 = requests.get(f'{URL}/scoring').text
response3 = requests.get(f'{URL}/summarystats').text
response4 = requests.get(f'{URL}/diagnostics').text

#combine all API responses
combined_response = {
    "predictions": json.loads(response1),
    **json.loads(response2),
    "summary_statistics": json.loads(response3),
    "diagnostics": json.loads(response4)
}

# Write the combined response to a .txt file
filepath = os.path.join(model_path,'apireturns.txt')
with open(filepath,'w') as f:
    f.write(json.dumps(combined_response))


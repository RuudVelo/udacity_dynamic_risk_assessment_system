import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df = pd.DataFrame()
    files = []

    for file in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file)
        print(file)
        print(file_path)
        df_input = pd.read_csv(file_path)
        print(df_input.head(2))
        file = os.path.join(*file_path.split(os.path.sep)[-3:])
        print(file)

if __name__ == '__main__':
    merge_multiple_dataframe()
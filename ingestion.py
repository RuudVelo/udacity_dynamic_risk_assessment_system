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
        df_input = pd.read_csv(file_path)
        df = df.append(df_input, ignore_index=True)

        files.append(file_path)

    df.drop_duplicates(inplace=True)
    
    # Write data to csv
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    # Write data with timestamp
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file: 
        file.write("\n".join(files))

if __name__ == '__main__':
    merge_multiple_dataframe()

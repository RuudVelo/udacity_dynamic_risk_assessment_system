import requests
import os
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])
filepath = os.path.join(test_data_path, "testdata.csv")

# Call each API endpoint and store the responses
response1 = requests.get(f"{URL}/prediction" + f"?filename={filepath}").text
response2 = requests.get(f"{URL}/scoring").text
response3 = requests.get(f"{URL}/summarystats").text
response4 = requests.get(f"{URL}/diagnostics").text

# combine all API responses
# Write the combined response to a .txt file
filepath = os.path.join(model_path, "apireturns2.txt")
with open(filepath, "w") as f:
    f.write("Model predictions\n")
    f.write(response1)
    f.write("\nModel pcore\n")
    f.write(response2)
    f.write("\nData Statistics\n")
    f.write(response3)
    f.write("\nModel diagnostics\n")
    f.write(response4)

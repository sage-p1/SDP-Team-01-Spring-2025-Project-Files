import pandas as pd 
import torch
from chronos import BaseChronosPipeline
import requests
import json
import os

def configure_models(time, target, datatype, filename, delimiter):
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
        device_map="cpu",  # use "cpu" for CPU inference
        torch_dtype=torch.bfloat16,
    )
    input_path = "static/input/" + filename

    if datatype == "csv":
        df = pd.read_csv(input_path, sep=delimiter)
    elif datatype == "json":
        df = pd.read_json(input_path)
    elif datatype == "xml":
        df = pd.read_xml(input_path)
    else:
        df = pd.read_excel(input_path)

    return forecast(pipeline, df, time, target, filename)

def forecast(pipeline, df, time, target, filename):
    prediction_length = 12
    # context must be either a 1D tensor, a list of 1D tensors,
    # or a left-padded 2D tensor with batch as the first dimension
    # quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
    # mean is an fp32 tensor with shape [batch_size, prediction_length]
    quantiles, mean = pipeline.predict_quantiles(
        context=torch.tensor(df[target]),
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

    # Convert the time column to datetime
    time_df = pd.to_datetime(df[time])
    # Calculate the time difference between consecutive times
    time_differences = time_df.diff().dropna()
    # Assume the pattern continues with the most common difference
    most_common_difference = time_differences.value_counts().idxmax()
    # Generate the next time
    last_time = time_df.iloc[-1]
    next_time = last_time + most_common_difference
    # Define how many next times you want (e.g., generate the next 5 times)
    next_times = [next_time + i * most_common_difference for i in range(0, prediction_length)]
    
    start = time_df.iloc[0]
    start = start.strftime("%Y-%m-%d %H:%M:%S")
    end = next_times[-1]
    end = end.strftime("%Y-%m-%d %H:%M:%S")

    data = {time:next_times, "Low": low.tolist(), "Median": median.tolist(), "High": high.tolist()}

    output_path = f"static/output/{filename.split(".")[0]}_output.csv"
    output_df = pd.DataFrame(data)
    output_df.to_csv(output_path, index=False)
    
    with open("output_path.txt", "w") as f:
        f.write(output_path)

    return post(filename, time, target, start, end)

def post(filename, time, target, start, end):
    grafana_url = "http://localhost:3000"
    username = "admin"  
    password = "admin"    

    response = requests.get(f"{grafana_url}/api/orgs", auth=(username, password))
    print("Response Status:", response.status_code)

    if response.status_code == 200:
        print("Authenticated successfully.")
        
        with open("backend/dashboard_payload.json") as f:
            dashboard_payload = json.load(f)

        dashboard_payload["dashboard"]["panels"][0]["targets"][0]["columns"][0]["selector"] = time
        dashboard_payload["dashboard"]["panels"][0]["targets"][0]["columns"][1]["selector"] = target 
        dashboard_payload["dashboard"]["panels"][0]["targets"][1]["columns"][0]["selector"] = time
        dashboard_payload["dashboard"]["time"]["from"] = start
        dashboard_payload["dashboard"]["time"]["to"] = end
        dashboard_payload["dashboard"]["panels"][0]["title"] = f"{filename} Panel"
        dashboard_payload["dashboard"]["title"] = f"{filename} Dashboard"

        # Create the dashboard using POST
        create_dashboard_response = requests.post(
            f"{grafana_url}/api/dashboards/db",
            json=dashboard_payload,
            auth=(username, password)
        )

        if create_dashboard_response.status_code == 200 or create_dashboard_response.status_code == 409:
            print("Dashboard created successfully.")
            print("Response:", create_dashboard_response.json())
        else:
            print("Failed to create dashboard.")
            print("Response:", create_dashboard_response.content)
    else:
        print("Failed to authenticate.")

    # Configuration
    token = "glsa_XX5cWW1iyQUUuCPYrq6pZRfisUiR9zgR_b1fed14a"
    uid = "test"

    # URL to fetch the dashboard
    dashboard_url = f"{grafana_url}/api/dashboards/uid/{uid}"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(dashboard_url, headers=headers)
    
    if response.status_code == 200:
        dashboard = response.json()['dashboard']
    else:
        raise ConnectionError(f"Failed to fetch dashboard: {response.status_code}, {response.text}")

    # URL for creating a snapshot
    snapshot_url = f"{grafana_url}/api/snapshots"

    # Payload to create a snapshot
    payload = {
        "dashboard": dashboard,
        "expire": 3600  # Optional: set expiration time in seconds
    }

    # Make the request to create a snapshot
    response = requests.post(snapshot_url, headers=headers, json=payload)

    if response.status_code == 200:
        snapshot_data = response.json()
        snapshot_url = snapshot_data.get('url')
        print("Snapshot created successfully!")
        print(f"Access your snapshot here: {snapshot_url}")
        return snapshot_url
    else:
        print("Failed to create snapshot")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

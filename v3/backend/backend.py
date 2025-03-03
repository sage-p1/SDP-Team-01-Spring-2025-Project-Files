import pandas as pd 
import torch
from chronos import BaseChronosPipeline
import requests
import json

def configure_models(time, target, prediction_length, sliding_window, datatype, filename, delimiter):
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
        device_map="cpu",  # use "cpu" for CPU inference
        torch_dtype=torch.bfloat16,
    )

    input_path = "static/input/" + filename

    # process input depending on datatype
    if datatype == "csv":
        df = pd.read_csv(input_path, sep=delimiter)
    elif datatype == "json":
        df = pd.read_json(input_path)
    elif datatype == "xml":
        df = pd.read_xml(input_path)
    else:
        df = pd.read_excel(input_path)

    return forecast(pipeline, df, time, target, filename, prediction_length, sliding_window, delimiter)

def forecast(pipeline, df, time, target, filename, prediction_length, sliding_window, delimiter):
    # partition df into 10 sliding windows
    if sliding_window:
        df_length = len(df[target])
        step = df_length//10
        # evenly space out 10 starting points across df
        starts = [i*step for i in range(10)]
        
        # 512 is the maximum context length for Chronos,
        # so we evenly space out 10 windows of length 512
        # if there is enough data to cover
        if df_length > 512*10: 
            ends = [start + 512 for start in starts]
        else:
            ends = [start + step for start in starts]
 
    else:
        # only need one window if sliding_window is False
        step = -1
        starts = [0]
        ends = [step]

    # this file contains the first start index (0), 
    # the step, the prediction length, and delimiter
    # so the input path in app.py knows what partition to select
    with open("input_step.txt", "w") as f:
        f.write("0\n")
        f.write(f"{step}\n") 
        f.write(f"{prediction_length}\n")
        f.write(f"{delimiter}")
        f.close()
    
    # build list of start and end timestamps
    start_times = []
    end_times = []

    # for each window
    for i in range(len(starts)):
        # context must be either a 1D tensor, a list of 1D tensors,
        # or a left-padded 2D tensor with batch as the first dimension
        # quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
        # mean is an fp32 tensor with shape [batch_size, prediction_length]
        quantiles, mean = pipeline.predict_quantiles(
            context=torch.tensor(df[target][starts[i]:ends[i]].values),
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

        # convert the time column to datetime
        time_df = pd.to_datetime(df[time][starts[i]:ends[i]])
        # calculate the time difference between consecutive times
        time_differences = time_df.diff().dropna()
        # assume the pattern continues with the most common difference
        most_common_difference = time_differences.value_counts().idxmax()
        # generate the next time
        last_time = time_df.iloc[-1]
        next_time = last_time + most_common_difference
        # define how many next times you want (e.g., generate the next 5 times)
        next_times = [next_time + i * most_common_difference for i in range(0, prediction_length)]
        
        # convert times to string for writing output
        start = time_df.iloc[0]
        start_time = start.strftime("%Y-%m-%d %H:%M:%S")
        start_times.append(start_time)

        end = next_times[-1]
        end_time = end.strftime("%Y-%m-%d %H:%M:%S")
        end_times.append(end_time)

        data = {time:next_times, "Low": low.tolist(), "Median": median.tolist(), "High": high.tolist()}

        output_path = f"static/output/{filename.split(".")[0]}_output{i}.csv"
        output_df = pd.DataFrame(data)
        output_df.to_csv(output_path, index=False)

    # write output path
    # this file tells app.py where to generally find the output
    with open("output_path.txt", "w") as f:
        f.write(f"static/output/{filename.split(".")[0]}_output.csv")
        f.close()
        
    # write current window to know which output file to display
    with open("window.txt", "w") as f: 
        f.write("0")
        f.close()

    # write start times
    with open("start_times.txt", "w") as f:
        for start_time in start_times:
            f.write(f"{start_time}\n")
        f.close()

    # write end times
    with open("end_times.txt", "w") as f:
        for end_time in end_times:
            f.write(f"{end_time}\n")
        f.close()

    # set times to the first window
    start_time = start_times[0]
    end_time = end_times[0]

    return post(filename, time, target, start_time, end_time)

def post(filename, time, target, start_time, end_time):
    grafana_url = "http://localhost:3000"
    username = "admin"  
    password = "admin"    

    # get request to Grafana
    response = requests.get(f"{grafana_url}/api/orgs", auth=(username, password))
    print("Response Status:", response.status_code)

    if response.status_code == 200:
        print("Authenticated successfully.")
        
        # payload is from a sample dashboard that keeps getting ovewritten
        with open("backend/dashboard_payload.json") as f:
            dashboard_payload = json.load(f)
            f.close()

        # update display fields
        dashboard_payload["dashboard"]["panels"][0]["targets"][0]["columns"][0]["selector"] = time
        dashboard_payload["dashboard"]["panels"][0]["targets"][0]["columns"][1]["selector"] = target 
        dashboard_payload["dashboard"]["panels"][0]["targets"][1]["columns"][0]["selector"] = time
        dashboard_payload["dashboard"]["time"]["from"] = start_time
        dashboard_payload["dashboard"]["time"]["to"] = end_time
        dashboard_payload["dashboard"]["panels"][0]["title"] = f"{filename.split(".")[0]} Panel"
        dashboard_payload["dashboard"]["title"] = f"{filename.split(".")[0]} Dashboard"

        # create the dashboard using POST
        create_dashboard_response = requests.post(
            f"{grafana_url}/api/dashboards/db",
            json=dashboard_payload,
            auth=(username, password)
        )

        # handle response
        if create_dashboard_response.status_code == 200 or create_dashboard_response.status_code == 409:
            print("Dashboard created successfully.")
            print("Response:", create_dashboard_response.json())
        else:
            print("Failed to create dashboard.")
            print("Response:", create_dashboard_response.content)
    else:
        print("Failed to authenticate.")

    # token for privileged groups to see
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

    # payload to create a snapshot
    payload = {
        "dashboard": dashboard,
        "expire": 3600 
    }

    # Make the request to create a snapshot
    response = requests.post(snapshot_url, headers=headers, json=payload)

    if response.status_code == 200:
        snapshot_data = response.json()
        snapshot_url = snapshot_data.get('url')
        print("Snapshot created successfully!")
        print(f"Access your snapshot here: {snapshot_url}")
        return snapshot_url, start_time, end_time
    else:
        print("Failed to create snapshot")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

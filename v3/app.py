from flask import Flask, render_template, request, jsonify
import os
from backend.backend import configure_models
import requests
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# home page
@app.route("/", methods=["GET","POST"])
def index():
    return home()

@app.route("/home", methods=["GET","POST"])
def home():  
    return render_template("home.html")

# intermediate webpage to initiate forecasts
@app.route("/result", methods=["GET", "POST"])
def result():
    submit = request.form.get("submit") # get submit data
    if submit: # if submit button was triggered
        time = request.form.get("time") # get time field
        target = request.form.get("target") # get target field

        # set optional parameters depending on what the user entered
        prediction_length = int(request.form.get("prediction_length")) if request.form.get("prediction_length") else 12
        sliding_window = request.form.get("sliding_window") if request.form.get("sliding_window") else False
        delimiter = request.form.get("delimiter") if request.form.get("delimiter") else ","

        filename = link = inline_text = ""

        # extensions dict
        extensions = {"csv": ".csv", "json": ".json", "xml": ".xml", "xlsx": ".xlsx", "sheets": ".xlsx", "excel": ".xlsx"}
        
        path = "static/input/"

        # determine which upload method was used and save input file
        method = request.form.get("upload_method")
        if method == "local": # upload from computer
            datatype = request.form.get("local_datatype")
            file = request.files["file"]
            filename = file.filename
            path += filename
            file.save(path)

        elif method == "url": # download from link
            datatype = request.form.get("url_datatype")
            link = request.form.get("link")
            filename += link.split("/")[-1]
            path += filename

            # establish connection and download data
            response = requests.get(link)
            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
                    f.close()
            
        else: # inline text
            datatype = request.form.get("inline_datatype")
            inline_text = request.form.get("inline_text")
            filename = request.form.get("inline_filename") if request.form.get("inline_filename") else "inline"
            filename += extensions[datatype]
            path += filename

            with open(path, "w") as f:
                for line in inline_text:
                    text = line
                    text.strip("\n")
                    f.write(text)
                    f.close()
        
        # write path to input file
        # the existence of this file determines if there is data Grafana can source
        with open("input_path.txt", "w") as f: 
            f.write(path)
            f.close()

        # run Chronos on input
        snapshot_url, start_time, end_time = configure_models(time, target, prediction_length, sliding_window, datatype, filename, delimiter)

        return render_template("result.html", output=1, snapshot_url=snapshot_url, sliding_window=sliding_window, start_time=start_time, end_time=end_time)
    return render_template("result.html")

# route to host input data source for Grafana
@app.route("/input", methods=["GET","POST"])
def input(): 
    if os.path.exists("input_path.txt"): # ensure file exists
        with open("input_path.txt", "r") as f: # get input file path
            path = f.read()
            f.close()

        with open("input_step.txt", "r") as f:
            start = int(f.readline()) # get start of index slice
            step = int(f.readline()) # get step between start and end
            prediction_length = int(f.readline()) # get prediction length
            delimiter = f.readline()
            f.close()

        # read input file depending on datatype
        if os.path.exists(path): 
            datatype = path.split(".")[-1] 
            if datatype == "csv":
                df = pd.read_csv(path, sep=delimiter)
            elif datatype == "json":
                df = pd.read_json(path)
            elif datatype == "xml":
                df = pd.read_xml(path)
            else:
                df = pd.read_excel(path)
        
        if step == -1: # no sliding windows, so display all data
            prediction_length = 0
        elif step > 512:
            step = 512
        return df[start:start+step+prediction_length].to_csv(index=False) # return as csv for parsing
    
    return "You must upload data before accessing this page"

# output data source for Grafana
@app.route("/output", methods=["GET","POST"])
def output(): 
    if os.path.exists("output_path.txt"):
        with open("output_path.txt", "r") as f: # get general output path
            path = f.read()
            f.close()

        with open("window.txt", "r") as f: # get current window number
            window = int(f.read())
            f.close()
        
        path = "".join([path.split(".")[0], str(window), ".csv"]) # get path of specific output at the ith window

        if os.path.exists(path): 
            with open(path, "r") as f: # get output file contents
                data = f.read()
                f.close()
        return data
    return "You must upload data before accessing this page"

# route for updating data in sliding window
@app.route("/left", methods=["GET", "POST"])
def left():
    result = {"message": "you must upload data before accessing this page", "start_time": "None", "end_time": "None"}   

    # ensure all files to read exist
    if os.path.exists("input_step.txt") and os.path.exists("window.txt") and os.path.exists("start_times.txt") and os.path.exists("end_times.txt"):
        with open("input_step.txt", "r") as f:
            start = int(f.readline())
            step = int(f.readline())
            prediction_length = int(f.readline())
            delimiter = f.readline()
            f.close()
        
        window = 0 
        if start > 0: # do not go beyond the first window
            start -= step # step back to previous window
            
            # update input_step file
            with open("input_step.txt", "w") as f:
                f.write(f"{start}\n")
                f.write(f"{step}\n")
                f.write(f"{prediction_length}\n")
                f.write(f"{delimiter}")
                f.close()

            # get window
            with open("window.txt", "r") as f:
                window = int(f.readline())
                f.close()
        
            # update window
            with open("window.txt", "w") as f:
                window -= 1
                f.write(f"{window}")
                f.close()

        # get start time
        with open("start_times.txt", "r") as f:
            i = -1
            while (i < window): # get date at the correct window
                start_time = f.readline().strip("\n")
                i += 1
            f.close()

        # get end time
        with open("end_times.txt", "r") as f:
            i = -1
            while (i < window): # get date at the correct window
                end_time = f.readline().strip("\n")
                i += 1
            f.close()

        result = {"message": "function successfully called", "start_time": start_time, "end_time": end_time}
    
    return jsonify(result)
    
# route for updating data in sliding window
@app.route("/right", methods=["GET","POST"])
def right():
    result = {"message": "you must upload data before accessing this page", "start_time": "None", "end_time": "None"}   

    # ensure all files to read exist
    if os.path.exists("input_step.txt") and os.path.exists("window.txt") and os.path.exists("start_times.txt") and os.path.exists("end_times.txt"):
        
        with open("window.txt", "r") as f:
            window = int(f.readline())
            f.close()
        
        if window < 9: # do not go past the last (tenth) window

            # update window
            with open("window.txt", "w") as f:
                window += 1
                f.write(f"{window}")
                f.close()
        
            # get input start, step, and pred_len
            with open("input_step.txt", "r") as f:
                start = int(f.readline())
                step = int(f.readline())
                prediction_length = int(f.readline())
                delimiter = f.readline()
                f.close()
            
            # move start forward to next window
            with open("input_step.txt", "w") as f:
                start += step
                f.write(f"{start}\n")
                f.write(f"{step}\n")
                f.write(f"{prediction_length}\n")
                f.write(f"{delimiter}")
                f.close()

        # get current start time
        with open("start_times.txt", "r") as f:
            i = -1
            while (i < window): # get date at the correct window
                start_time = f.readline().strip("\n")
                i += 1
            f.close()

        # get current end time
        with open("end_times.txt", "r") as f:
            i = -1
            while (i < window): # get date at the correct window
                end_time = f.readline().strip("\n")
                i += 1
            f.close()

        result = {"message": "function successfully called", "start_time": start_time, "end_time": end_time}
    
    return jsonify(result)

# how it works page
@app.route("/how_it_works", methods=["GET","POST"])
def how_it_works():
    return render_template("how_it_works.html")  

# project background page
@app.route("/background", methods=["GET","POST"])
def background():
    return render_template("background.html")

# team page
@app.route("/team", methods=["GET","POST"])
def team():
    return render_template("team.html")

# video page
@app.route("/video", methods=["GET","POST"])
def video():
    return render_template("video.html")

# references page
@app.route("/references", methods=["GET","POST"])
def references():
    return render_template("references.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
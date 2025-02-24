from flask import Flask, render_template, request
import os
from backend.backend import configure_models
import requests

app = Flask(__name__)

# home page
@app.route("/", methods=["GET","POST"])
def index():
    return home()

@app.route("/home", methods=["GET","POST"])
def home():  
    return render_template("home.html")

# intermediate webpage for 
@app.route("/result", methods=["GET", "POST"])
def result():
    submit = request.form.get("submit")
    if submit:
        time = request.form.get("time")
        target = request.form.get("target")

        filename = link = inline_text = ""

        extensions = {"csv": ".csv", "json": ".json", "xml": ".xml", "xlsx": ".xlsx", "sheets": ".xlsx", "excel": ".xlsx"}
        
        path = "static/input/"

        # determine which upload method was used and save input file
        method = request.form.get("upload_method")
        if method == "local":
            datatype = request.form.get("local_datatype")
            
            file = request.files["file"]
            filename = file.filename
            path += filename
            file.save(path)

        elif method == "url":
            datatype = request.form.get("url_datatype")
            link = request.form.get("link")
            filename += link.split("/")[-1]
            path += filename

            response = requests.get(link)

            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
            
        else:
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
        
        delimiter = request.form.get("delimiter") if request.form.get("delimiter") else ","
        
        with open("input_path.txt", "w") as f: # write path to input file
            f.write(path)

        snapshot_url = configure_models(time, target, datatype, filename, delimiter)

        print("mark", snapshot_url)
        return render_template("result.html", output=1, snapshot_url=snapshot_url)
    return render_template("result.html")

@app.route("/input", methods=["GET","POST"])
def input(): # link for Grafana to grab data
    if os.path.exists("input_path.txt"): 
        with open("input_path.txt", "r") as f: # get input file path
            path = f.read()
            f.close()
        if os.path.exists(path): 
            with open(path, "r") as f: # get input file contents
                data = f.read()
                f.close()
        return data
    return "You must upload data before accessing this page"

@app.route("/output", methods=["GET","POST"])
def output(): 
    if os.path.exists("output_path.txt"): 
        with open("output_path.txt", "r") as f: # get output file path
            path = f.read()
            f.close()
        if os.path.exists(path): 
            with open(path, "r") as f: # get input file contents
                data = f.read()
                f.close()
        return data
    return "You must upload data before accessing this page"

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

# references page
@app.route("/video", methods=["GET","POST"])
def video():
    return render_template("video.html")

# references page
@app.route("/references", methods=["GET","POST"])
def references():
    return render_template("references.html")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from backend.chronos_test import chronos_test
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'team01'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home(): # allows users to upload data from 
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data #  grab the file
        filename = file.filename # get filename
        data = False 
        site_output = request.form.to_dict()
        time = site_output["time"] # get time data field
        target = site_output["target"] # get target data field
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(filename))) # Then save the file
        with open("path.txt", "w") as f : # write path to input file
            f.write("static/files/" + filename)
        return render_template('generate.html', filename=filename, data=data, time=time, target=target)
    return render_template('index.html', form=form)

@app.route('/generate', methods=["GET", "POST"])
def generate():
    # grab text input
    site_output = request.form.to_dict()
    if "data" in site_output:
        data = site_output["data"] # get raw data
        with open("static/files/input.csv", "w") as f: # save in input.txt
            f.write(data)
        filename = False
        site_output = request.form.to_dict()
        time = site_output["time"] # get time data field
        target = site_output["target"] # get target file name
        with open("path.txt", "w") as f : # write path to input file
            f.write("static/files/input.csv")
        return render_template('generate.html', filename=filename, data=data, time=time, target=target)
    return render_template('generate.html')

@app.route('/result', methods=["GET", "POST"])
def result():
    site_output = request.form.to_dict()
    filename = "input.csv"
    target = button1 = button2 = False
    if "button1" in site_output:
        button1 = True
    
    if "button2" in site_output:
        button2 = True

    if "time" in site_output:
        time = site_output["time"]

    if "target" in site_output:
        target = site_output["target"]
    
    if "filename" in site_output:
        filename = site_output["filename"]

    if button1 or button2:
        chronos_test(filename, time, target)
    return render_template('result.html', button1=button1, button2=button2, target=target, filename=filename)

@app.route('/input', methods=['GET',"POST"])
def input(): # link for Grafana to grab data
    if os.path.exists("path.txt"): 
        with open("path.txt", "r") as f: # get input file path
            path = f.read()
            f.close()
        if os.path.exists(path): 
            with open(path, "r") as f: # get input file contents
                data = f.read()
                f.close()
        return data
    return "You must upload data before accessing this page"

@app.route('/output', methods=['GET',"POST"])
def output(): # allows users to upload data from 
    if os.path.exists("static/files/output.csv"): 
        with open("static/files/output.csv", "r") as f: # get output file path
            data = f.read()
            f.close()
        return data
    return "You must upload data and generate results before accessing this page"

if __name__ == '__main__':
    app.run(debug=True)
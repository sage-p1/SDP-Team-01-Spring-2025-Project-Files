from flask import Flask, render_template, request, jsonify
import os
import torch
from backend.backend import configure_models
import requests
from flask_cors import CORS
import pandas as pd
import numpy as np
from backend.data_provider.data_factory import data_provider
from backend.utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from backend.models.PatchTST import PatchTST
from backend.models.GPT4TS import GPT4TS
from backend.models.DLinear import DLinear
import argparse
from werkzeug.utils import secure_filename

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
        chronos_model = request.form.get("chronos_model")

        # set optional parameters depending on what the user entered
        prediction_length = int(request.form.get("prediction_length")) if request.form.get("prediction_length") else 12
        num_windows = int(request.form.get("num_windows")) if request.form.get("num_windows") else 10
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
        snapshot_url, start_time, end_time = configure_models(time, target, prediction_length, num_windows, datatype, filename, chronos_model, delimiter)

        return render_template("result.html", output=1, snapshot_url=snapshot_url, start_time=start_time, end_time=end_time)
    return render_template("result.html")

# route to host input data source for Grafana
@app.route("/chronos-input", methods=["GET","POST"])
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
        
        if step > 512:
            step = 512
        return df[start:start+step+prediction_length].to_csv(index=False) # return as csv for parsing
    
    return "You must upload data before accessing this page"

# output data source for Grafana
@app.route("/chronos-output", methods=["GET","POST"])
def output(): 
    if os.path.exists("output_path.txt"):
        with open("output_path.txt", "r") as f: # get general output path
            path = f.read()
            f.close()

        with open("window.txt", "r") as f: # get current window number
            window = int(f.readline())
            f.close()
        
        path = "".join([path.split(".")[0], "chronos", str(window), ".csv"]) # get path of specific output at the ith window

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
                num_windows = int(f.readline())
                f.close()
        
            # update window
            with open("window.txt", "w") as f:
                window -= 1
                f.write(f"{window}\n{num_windows}")
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
            num_windows = int(f.readline())
            f.close()
        
        if window < num_windows - 1: # do not go past the last window

            # update window
            with open("window.txt", "w") as f:
                window += 1
                f.write(f"{window}\n{num_windows}")
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

@app.route("/upload", methods=["GET", "POST"])
def upload():
    return render_template("upload_data.html")

@app.route('/ofa', methods = ['GET', 'POST'])
def ofa_home():
    return render_template("ofa.html")

# OFA functions
def train_and_evaluate(args):
    mses, maes = [], []
    args.pred_len = int(args.pred_len)


    for ii in range(args.itr):
        ii = int(ii)


        setting = f"{args.model_id}_sl{336}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_gl{args.gpt_layers}_df{args.d_ff}_eb{args.embed}_itr{ii}"
        path = os.path.join(args.checkpoints, setting)
        checkpoint_file = os.path.join(path, 'checkpoint.pth')

        os.makedirs(path, exist_ok=True)

        train_data, train_loader = data_provider(args, 'train')
        args.enc_in = train_data.enc_in  # This pulls the value set in __read_data__
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.model == 'PatchTST':
            model = PatchTST(args, device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
        else:
            model = GPT4TS(args, device)

        model.to(device)

        if os.path.exists(checkpoint_file):
            print(f"📦 Loading pretrained model from: {checkpoint_file}")
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        else:
            print(f"🚀 Training model because checkpoint not found at: {checkpoint_file}")
            model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            criterion = torch.nn.MSELoss()
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

            for epoch in range(args.train_epochs):
                train_loss = []
                for batch_x, batch_y, _, _ in train_loader:
                    model_optim.zero_grad()
                    batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
                    outputs = model(batch_x, ii)
                    outputs = outputs[:, -args.pred_len:, :]
                    batch_y = batch_y[:, -args.pred_len:, :]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    model_optim.step()
                    train_loss.append(loss.item())

                vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
                scheduler.step() if args.cos else adjust_learning_rate(model_optim, epoch + 1, args)
                early_stopping(vali_loss, model, path)
                if early_stopping.early_stop:
                    break

            model.load_state_dict(torch.load(checkpoint_file, map_location=device))

        mse, mae = test(model, test_data, test_loader, args, device, ii)
        mses.append(mse)
        maes.append(mae)

    return {
    "mse_mean": float(np.mean(mses).item()),
    "mse_std": float(np.std(mses).item()),
    "mae_mean": float(np.mean(maes).item()),
    "mae_std": float(np.std(maes).item())
}


@app.route('/train', methods=['POST'])
def train_model():
    data = request.json

    if 'file_path' not in data:
        return jsonify({"error": "No file path provided. Please upload a file first."}), 400


    filename = data['file_path'].replace('uploads/', '')
    full_path = os.path.join("uploads", filename)

    #  Debugging: Print to check correct file path
    print(f"Checking file path: {full_path}")

    # if full_path.startswith('uploads/'):
    #     full_path = full_path.replace('uploads/', '')

     # Check if the 'uploads' directory exists
    if not os.path.exists("uploads"):
        print("⚠️ 'uploads' directory is missing!")
        return jsonify({"error": "Uploads folder is missing!"}), 400

    # Check if the file exists
    # full_path = os.path.join("uploads", file_path)

    if not os.path.exists(full_path):
        print(f"⚠️ File '{full_path}' not found in 'uploads' directory!")
        return jsonify({"error": f"File '{filename}' not found. Ensure it was uploaded correctly."}), 400

    required_defaults = {
        "model_id": os.path.basename(filename).replace(".csv", ""),
        "label_len": 168,  # Default value (adjust if needed)
        "checkpoints": "checkpoints",
        "data_path": filename,  # Store actual file path
       # "data_path": os.path.join("uploads", os.path.basename(file_path)),  # Ensure correct path
        "root_path": "uploads",  # Root directory where files are stored
        "data": "custom",  # Path to dataset (needed by data_provider)
        "pred_len": 96,  # Prediction length (default, adjust if needed)
        "d_model": 768,  # Model dimension size
        "n_heads": 4,  # Number of attention heads
        "e_layers": 3,  # Encoder layers
        "gpt_layers": 3,  # GPT layers
        "d_ff": 768,  # Feed-forward network dimension
        "embed": "timeF",  # Embedding type
        "itr": 1,  # Number of training iterations
        "train_epochs": 10,  # Default training epochs
        "learning_rate": 0.0001,  # Default learning rate
        "patience": 5,  # Early stopping patience
        "tmax": 20,  # CosineAnnealingLR max decay steps
        "cos": 1,  # Use cosine learning rate scheduler
        "model": "GPT4TS",  # Default model type
        "percent": 1.0,  # 🔥 Add 'percent' to prevent the AttributeError
        "max_len": 512,  # 🔥 Add 'max_len' to match data_provider
        "batch_size": 32,  # Add batch_size (if used in DataLoader)
        "num_workers": 4,  # Add num_workers for DataLoader
        "seq_len": 336,  # Sequence length for training
        "features": "M",  # Feature type (modify as needed)
        "target": "OT",  # Target column (modify as needed)
        "freq": "h",  # Frequency of data (modify as needed)
        "enc_in": 1,  # Default to 1 for univariate, set to # of features for multivariate
        "c_out" : 1,
        "patch_size" : 16,
        "stride" : 8,
        "dropout" : 0.2,

    }

    # Merge user-provided data with default values
    for key, value in required_defaults.items():
        if key not in data:
            data[key] = value

    args = argparse.Namespace(**data)

    # Get enc_in from data
    train_data, _ = data_provider(args, 'train')
    args.enc_in = train_data.enc_in
    print(f"Determined enc_in: {args.enc_in}")

    results = train_and_evaluate(args)
    return jsonify(results)

    print("Received data:", data)




if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
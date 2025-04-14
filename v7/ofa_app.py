from flask import Flask, request, jsonify, render_template
import torch
import os
import numpy as np
from backend.data_provider.data_factory import data_provider
from backend.utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from backend.models.PatchTST import PatchTST
from backend.models.GPT4TS import GPT4TS
from backend.models.DLinear import DLinear
import argparse
import logging
from werkzeug.utils import secure_filename

#Flask web app
app = Flask(__name__)

#directory for storing files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        print("Received a POST request")  # 🔍 Debugging


        if 'file' not in request.files:
            print("No file in request!")
            return render_template("home.html", message = "No file selected. Please try again.")

        file = request.files['file']
        print(f"File received: {file.filename}")

        if file.filename =='':
            print("Empty filename received!")
            return render_template("home.html", message = "No file selected. Please try again.")

        if file and file.filename.endswith(".csv"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            print(f"Saving file to: {file_path}")
            file.save(file_path)
            print(f"File saved at: {file_path}")  # 🔍 Debugging
          #  return render_template("index.html", message = "File upload successfully!")




            if os.path.exists(file_path):
                print(f"File successfully saved at: {file_path}")  # 🔍 Debugging
            else:
                print("File did not save correctly!")

            return render_template("home.html", message="File uploaded successfully!")

    return render_template("home.html")
    return 'Welcome to the Flask API for Long-Term Forecasting!'


def train_and_evaluate(args):
    mses, maes = [], []
    args.pred_len = int(args.pred_len)


    for ii in range(args.itr):
        ii = int(ii)


        setting = f"{args.model_id}_sl{336}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_gl{args.gpt_layers}_df{args.d_ff}_eb{args.embed}_itr{ii}"
        path = os.path.join(args.checkpoints, setting)
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

        model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        for epoch in range(args.train_epochs):
            train_loss = []

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()
                batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)

                #args.pred_len = int(args.pred_len)


                assert batch_x.dtype == torch.float32, f"batch_x must be float32, got {batch_x.dtype}"
                assert batch_y.dtype == torch.float32, f"batch_y must be float32, got {batch_y.dtype}"
                assert not torch.isnan(batch_x).any(), "NaN values found in batch_x!"
                assert not torch.isnan(batch_y).any(), "NaN values found in batch_y!"


        # Check tensor dimensions for outputs and batch_y
                # print(f"outputs shape: {outputs.shape}")
                # print(f"batch_y shape: {batch_y.shape}")

                # outputs = model(batch_x, ii)[:, -args.pred_len:, :]
                # print(f"Model outputs shape: {outputs.shape}")  # Debugging
                # print(f"batch_x shape: {batch_x.shape}")
                # print(f"batch_y shape: {batch_y.shape}")

    # Ensure batch_x and batch_y are 3D tensors
                if len(batch_x.shape) != 3 or len(batch_y.shape) != 3:
                    print(f"Unexpected shape for batch_x: {batch_x.shape}")
                    print(f"Unexpected shape for batch_y: {batch_y.shape}")

                # print(f"batch_x type: {type(batch_x)}, shape: {batch_x.shape}")
                # print(f"batch_y type: {type(batch_y)}, shape: {batch_y.shape}")
                # print(f"args.pred_len: {args.pred_len}, type: {type(args.pred_len)}")

                outputs = model(batch_x, ii)

    # Debugging: Print the shape of outputs
                # print(f"outputs shape before slicing: {outputs.shape}")
                pred_len_int = int(args.pred_len)
                outputs = outputs[:, -pred_len_int:, :]
                batch_y = batch_y[:, -pred_len_int:, :]
                # print(f"outputs shape after slicing: {outputs.shape}")
                # print(f"batch_y shape after slicing: {batch_y.shape}")

                loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            adjust_learning_rate(model_optim, epoch + 1, args) if not args.cos else scheduler.step()
            early_stopping(vali_loss, model, path)

            if early_stopping.early_stop:
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        model.load_state_dict(torch.load(best_model_path))

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
        print("'uploads' directory is missing!")
        return jsonify({"error": "Uploads folder is missing!"}), 400

    # Check if the file exists
    # full_path = os.path.join("uploads", file_path)

    if not os.path.exists(full_path):
        print(f"File '{full_path}' not found in 'uploads' directory!")
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


if __name__ == '__main__':
    app.run(debug=True)
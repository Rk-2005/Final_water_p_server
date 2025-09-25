import os
import logging
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------------------------
# Setup
# ------------------------------
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ------------------------------
# Load pretrained model + scaler
# ------------------------------
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

try:
    global_model = load_model(MODEL_PATH)
    global_scaler = joblib.load(SCALER_PATH)
    logging.info("✅ Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model/scaler: {e}")
    global_model, global_scaler = None, None

# ------------------------------
# Helper functions
# ------------------------------
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean NaNs and duplicates."""
    data.fillna(data.mean(), inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def prepare_features(data: pd.DataFrame):
    """Drop target column if present."""
    try:
        return data.drop(columns=["Potability"])
    except KeyError:
        return data  # assume file is test data

# ------------------------------
# API Routes
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    file.save(file_path)

    try:
        # Load CSV
        df = pd.read_csv(file_path)
        df = preprocess_data(df)
        X = prepare_features(df)

        if global_scaler is None or global_model is None:
            return jsonify({"error": "Model or scaler not loaded"}), 500

        # Transform + Predict
        scaled_X = global_scaler.transform(X)
        preds = global_model.predict(scaled_X).flatten().tolist()

        # Clean up
        os.remove(file_path)

        return jsonify({"potability_prediction": preds})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# ------------------------------
# Run Server
# ------------------------------
if __name__ == "__main__":
    app.run(port=5200, debug=True)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import zipfile
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Initialize FastAPI app
app = FastAPI()

# CORS configuration - allows requests from all origins (can be restricted later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Extract the zipped Random Forest model if needed
with zipfile.ZipFile('random_forest_model.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

# Load GNN (Deep Learning) model
gnn_model = keras.models.load_model("gnn.keras")

# Load Random Forest Regression model
with open('random_forest_model.sav', 'rb') as file:
    reg_model = pickle.load(file)

# Load StandardScaler for input normalization
with open('scaler.sav', 'rb') as file:
    scaler = pickle.load(file)

# Define expected input format using Pydantic
class InputData(BaseModel):
    runs: int
    wickets: int
    over: float
    striker: int
    nonStriker: int

# Utility function to preprocess input data
def preprocess_input(data: InputData):
    input_array = np.array([[data.runs, data.wickets, data.over, data.striker, data.nonStriker]])
    scaled_input = scaler.transform(input_array)
    return scaled_input

# Endpoint for prediction using Random Forest model
@app.post("/predict")
def predict_with_random_forest(data: InputData):
    """
    Predict final cricket score using Random Forest model.
    """
    processed_data = preprocess_input(data)
    prediction = reg_model.predict(processed_data)
    return {"predicted_final_score": prediction[0]}

# Endpoint for prediction using Deep Learning (GNN) model
@app.post("/predict-dll")
def predict_with_gnn(data: InputData):
    """
    Predict final cricket score using Deep Learning model (GNN).
    """
    processed_data = preprocess_input(data)
    prediction = gnn_model.predict(processed_data)
    return {"predicted_final_score": int(prediction[0][0])}

# To run the server, use the command: uvicorn filename:app --reload

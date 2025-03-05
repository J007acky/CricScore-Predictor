from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import zipfile

with zipfile.ZipFile('random_forest_model.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


# Load the trained RandomForest model
with open('random_forest_model.sav', 'rb') as file:
    reg = pickle.load(file)

# Load the trained StandardScaler
with open('scaler.sav', 'rb') as file:
    sc = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()


# Define request body format
class InputData(BaseModel):
    runs: int
    wickets: int
    over: float
    striker: int
    nonStriker: int


@app.post("/predict")
def predict_score(data: InputData):
    # Convert input to numpy array and standardize it
    print(data)
    runs = int(data.runs)
    wickets = int(data.wickets)
    overs = float(data.over)
    striker = int(data.striker)
    nonStriker = int(data.nonStriker)
    test_data = np.array([[runs, wickets, overs, striker, nonStriker]])
    print(test_data)
    test_data = sc.transform(test_data)

    # Make prediction
    prediction = reg.predict(test_data)
    print(prediction)
    return {"predicted_final_score": prediction[0]}

# Run the API using: uvicorn filename:app --reload

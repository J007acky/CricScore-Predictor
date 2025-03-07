from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import zipfile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras


app = FastAPI()


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:52168",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



with zipfile.ZipFile('random_forest_model.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


gnn_model = keras.models.load_model("gnn.keras")



# Load the trained RandomForest model
with open('random_forest_model.sav', 'rb') as file:
    reg = pickle.load(file)

# Load the trained StandardScaler
with open('scaler.sav', 'rb') as file:
    sc = pickle.load(file)


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


@app.post("/predict-dll")
def predict_score(data: InputData):
    print(data)
    runs = int(data.runs)
    wickets = int(data.wickets)
    overs = float(data.over)
    striker = int(data.striker)
    nonStriker = int(data.nonStriker)
    test_data = np.array([[runs, wickets, overs, striker, nonStriker]])
    print(test_data)
    test_data = sc.transform(test_data)
    prediction = gnn_model.predict(test_data)
    print(prediction[0][0])
    return {"predicted_final_score": int(prediction[0][0])}


# Run the API using: uvicorn filename:app --reload

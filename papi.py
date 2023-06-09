from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("C:\\Users\\Kshit\\Desktop\\Hackathon\\transactions_model2.pkl")  # Load your trained machine learning model

@app.post("/process_csv")

async def process_csv(file: UploadFile = File(...)):
    # Read the CSV file into a pandas DataFrame
    print("this is starting")
    inp = pd.read_csv(file.file)

    # Preprocessing
    # Specify the columns to be deleted
    

    # Make predictions using your trained model
    predictions = model.predict(inp)
    print(predictions)

 
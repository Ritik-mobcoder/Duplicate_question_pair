import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from Pipeline import test_pipline
import pandas as pd
import psycopg2
from DB_details import db_params

app = FastAPI()

import psycopg2


def save_data(question1, question2, result):
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()
    query = """INSERT INTO Prediction (question1, question2, results) 
                   VALUES ('%s', '%s', %s)"""
    cursor.execute(query % (question1, question2, result))
    connection.commit()
    cursor.close()
    connection.close()
    print("Data saved successfully")


@app.post("/predict")
async def root(request: Request):
    # try:
    body = await request.json()
    question1 = body["question1"]
    question2 = body["question2"]

    input_data = pd.DataFrame(
        {"question1": question1, "question2": question2}, index=[0]
    )
    pro = test_pipline(input_data)

    model_path = "model.joblib"
    model = joblib.load(model_path)
    prediction = model.predict(pro)
    save_data(question1, question2, result=prediction[0])
    response_data = {"prediction": str(prediction)}
    return JSONResponse(response_data)


# except Exception as e:
#     error_message = str(e)
#     response_data = {"error": error_message}
#     return JSONResponse(response_data)

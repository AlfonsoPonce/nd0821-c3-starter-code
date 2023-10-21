from typing import Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import joblib
from starter.ml.data import process_data

save_folder = './model/'


app = FastAPI()


# Declare the data object with its components and their type.
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
async def say_hello():
    return "Welcome to model API"


@app.post("/inference/")
async def ingest_data(inference: InputData):
    data = {  'age': inference.age,
                'workclass': inference.workclass,
                'fnlgt': inference.fnlgt,
                'education': inference.education,
                'education-num': inference.education_num,
                'marital-status': inference.marital_status,
                'occupation': inference.occupation,
                'relationship': inference.relationship,
                'race': inference.race,
                'sex': inference.sex,
                'capital-gain': inference.capital_gain,
                'capital-loss': inference.capital_loss,
                'hours-per-week': inference.hours_per_week,
                'native-country': inference.native_country,
                }

    # prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    if os.path.isfile(os.path.join(save_folder, 'trained_model.pkl')):
        model = joblib.load(open(os.path.join(save_folder, 'trained_model.pkl'), "rb"))
        encoder = joblib.load(open(os.path.join(save_folder, 'encoder.pkl'), "rb"))
        lb = joblib.load(open(os.path.join(save_folder, 'lb.pkl'), "rb"))

    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # get model prediction which is a one-dim array like [1]
    prediction = model.predict(sample)

    # convert prediction to label and add to data output
    if prediction[0] > 0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K',
    data['prediction'] = prediction

    return data
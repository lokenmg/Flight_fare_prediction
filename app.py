from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title = 'Milk Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

model = load(pathlib.Path('model/milk_prediction-v1.joblib'))

class InputData(BaseModel):
    pH:float=6.6
    Temprature:int=70
    Taste:int=1
    Odor:int=1
    Fat :int=1
    Turbidity:int=1
    Colour:int=255
    

class OutputData(BaseModel):
    score:float=0.9952830188679245

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict_proba(model_input)[:,-1]

    return {'score':result}

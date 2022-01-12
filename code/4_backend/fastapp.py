from fastapi import FastAPI
from pydantic import BaseModel
#from fastapi.encoders import jsonable_encoder
#from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import Readabilitymodel,FINAL_MODEL_PATH

model=Readabilitymodel(model_name='roberta-base',model_dir=FINAL_MODEL_PATH)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputExcerpt(BaseModel):
    excerpt:str

class Result(InputExcerpt):
    score:float
    level:int

@app.get("/ping")
async def ping():
    # return our data
    return {'ping': "I got you!"}

@app.post("/predict",response_model=Result,status_code=200)
async def get_prediction(item: InputExcerpt):
    excerpt=item.excerpt
    score,level=model.predict(excerpt)
    response_object = {"excerpt":excerpt,"score": score, "level": level}
    #print("respoinse_object is ",response_object)
    return response_object



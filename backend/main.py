import sys
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.predict import predict
from core.scraper import scraper


app = FastAPI()

class ArticleInput(BaseModel):
    article_url: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def read_predict(request: Request, payload: ArticleInput):
    url = payload.article_url
    title, txt = scraper(url)
    prediction = predict(title, txt)
    return {"prediction": prediction}
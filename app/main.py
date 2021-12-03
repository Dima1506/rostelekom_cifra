from typing import Optional
import requests
from fastapi import FastAPI

import gensim
import nltk
import pandas as pd
import re
import urllib.request
import numpy as np
import pymorphy2
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

app = FastAPI()


@app.get("/login")
def login():
    return {"url": "https://tvscp.tionix.ru/realms/master/protocol/openid-connect/auth?response_type=code&grant_type=authorization_code&client_id=tvscp&scope=openid&redirect_uri=http:/localhost:3000"}


@app.post("/get_token")
def get_token(code: str):
    url = 'https://tvscp.tionix.ru/realms/master/protocol/openid-connect/token'
    payload = {
        'client_id': 'tvscp',
        'client_secret': 'f3e94369-53ac-43d5-842e-09fe6d8a71ff',
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri':'http:/localhost:3000'
    }

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    r = requests.post(url, data=payload, headers=headers)
    #print(r.text)
    token = eval(r.text)['access_token']
    print(token)
    return {"token": token}

@app.post("/info_user")
def info_user(token: str):
    url = 'https://tvscp.tionix.ru/realms/master/protocol/openid-connect/userinfo'
    headers={'Authorization':'Bearer '+str(token)}
    r = requests.post(url, headers=headers)
    return eval(r.text)

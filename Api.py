from fastapi import FastAPI

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import User

from pydantic import BaseModel

app = FastAPI()

DB_CERT = 'roldo_DB.json'
cred = credentials.Certificate(DB_CERT)
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.get("/")
def home():
    return {"Data":"Test"}


@app.post("/register/")
def add_user(user: User):
    db.collection(u'users').document(f'{user.email}').set(user)
    return user




#uvicorn Api:app --reload


import Login
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import User


# from pydantic import BaseModel
import datetime
from uuid import uuid4
from google.cloud.firestore_v1.field_path import FieldPath
# import firestore as f

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CERT = 'roldo_DB.json'
cred = credentials.Certificate(DB_CERT)
firebase_admin.initialize_app(cred)
db = firestore.client()

KEYS={}

@app.get("/")
def home():
    return {"Data":"Test"}

@app.post("/user/register")
def register(user:User.User):
    db_user = db.collection(u'users').document(f'{user.email}').get()
    if not db_user.exists:
        db.collection(u'users').document(f'{user.email}').set({
            'password': user.password,
            'email': user.email,
            'name': user.name,})
        return True
    else:
        return False

@app.post("/user/login")
def login(login:Login.Login):
    user = db.collection(u'users').document(f'{login.email}').get()
    if user.exists and user.to_dict()['password'] == login.password:
        tok=generate_token()
        KEYS[login.email] = tok
        return tok
    else:
        return False

def check_token(token):
    for key in KEYS:
        if token == KEYS[key]:
            return True
    return False
def get_email_from_token(token):
    for key in KEYS:
        if token == KEYS[key]:
            return key
    return False
def generate_token():
    return str(uuid4())
def get_actual_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def save_emotion_data(user_email, emotion_data):
    db.collection(u'users').document(f'{user_email}').collection(u'emotions').document(get_actual_time()).set(emotion_data)

@app.post("/user/add_message")
def add_message(message,token):
    if check_token(token):
        user_email = get_email_from_token(token)
        db.collection(u'users').document(f'{user_email}').collection(u'messages').document(generate_token()).set({'data':message})
        return True
    else:
        return False
@app.post("/user/edit_message")
def edit_message(id,message,token):
    if check_token(token):
        user_email = get_email_from_token(token)
        db.collection(u'users').document(f'{user_email}').collection(u'messages').document(id).update({'data':message})
        return True
    else:
        return False
@app.get("/user/get_messages")
def get_messages(token):
    if check_token(token):
        user_email = get_email_from_token(token)
        messages = db.collection(u'users').document(f'{user_email}').collection(u'messages').stream()
        out={}
        for doc in messages:
            out[doc.id]=doc.to_dict()
        return out
    else:
        return False

# @app.post("/user/edit_message")

@app.post("/user/add_to_journal")
def add_to_journal(data,token):
    if check_token(token):
        email = get_email_from_token(token)
        db.collection(u'users').document(f'{email}').collection(u'journal').document(get_actual_time()).set({"data":data})
        return True
    else:
        return False

@app.get("/user/get_journal")
def get_journal(token):
    if check_token(token):
        email=get_email_from_token(token)
        # journal = db.collection(u'users').document(f'{email}').collection(u'journal').order_by(FieldPath.document_id(),direction=firestore.Query.DESCENDING).stream()
        journal = db.collection(u'users').document(f'{email}').collection(u'journal').stream()
        out={}
        for doc in journal:
            out[doc.id]=doc.to_dict()
        return out
    else:
        return False

@app.get("/user/destroy_token")
def destroy_token(token):
    for key in KEYS:
        if token == KEYS[key]:
            KEYS.pop(key)
            return True
    return False

@app.get("/user/recover_token")
def recover_token(user_email):
    if user_email in KEYS:
        return KEYS[user_email]
    else:
        return False


@app.websocket("/emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    token = await websocket.receive_text()
    if check_token(token):
        while True:
            data = await websocket.receive_text()
            save_emotion_data(data)
            # await websocket.send_text(f"Message text was: {data}")
    else:
        await websocket.close()
        return False

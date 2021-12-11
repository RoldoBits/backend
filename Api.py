#uvicorn Api.py:app --reload


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
    user = db.collection(u'users').document(f'{user.email}').get()
    if user.to_dict() == None:
        db.collection(u'users').document(f'{user.email}').set({
            'password': user.password,
            'email': user.email,
            'name': user.name,})
        return True
    else:
        return False

@app.post("/user/login")
def login(email, password):
    user = db.collection(u'users').document(f'{email}').get()
    if user.to_dict()['password'] == password:
        tok=generate_token()
        KEYS[email] = tok
        return tok
    else:
        return False

def check_token(token):
    for key in KEYS:
        if token == KEYS[key]:
            return True
    return False
def generate_token():
    return str(uuid4())
def get_actual_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def save_emotion_data(user_email, emotion_data):
    db.collection(u'users').document(f'{user_email}').collection(u'emotions').document(get_actual_time()).set(emotion_data)

@app.post("/user/send_message")
def send_message(user_email, message,token):
    if check_token(token):
        db.collection(u'users').document(f'{user_email}').collection(u'messages').document(get_actual_time()).set(message)
    else:
        return False
@app.get("/user/get_messages")
def get_messages(user_email,token):
    if check_token(token):
        messages = db.collection(u'users').document(f'{user_email}').collection(u'messages').get()
        return messages
    else:
        return False

@app.get("/user/add_to_journal")
def add_to_journal(user_email, journal_data,token):
    if check_token(token):
        db.collection(u'users').document(f'{user_email}').collection(u'journal').document(get_actual_time()).set(journal_data)
        return False
    else:
        return False

@app.get("/user/get_journal")
def get_journal(user_email,token):
    if check_token(token):
        journal = db.collection(u'users').document(f'{user_email}').collection(u'journal').get()
        return journal
    else:
        return False

@app.get("/user/destroy_token")
def destroy_token(token):
    for key in KEYS:
        if token == KEYS[key]:
            KEYS.pop(key)
            return True
    return False
 
@app.websocket("/emotion")
async def websocket_endpoint(websocket: WebSocket,token):
    if check_token(token):
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            save_emotion_data(data)
            # await websocket.send_text(f"Message text was: {data}")
    else:
        await websocket.close()
        return False

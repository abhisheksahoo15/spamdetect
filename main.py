from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


import nltk


# For cloud deployments where nltk_data is bundled locally
nltk.data.path.append("./nltk_data")

# Load your model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# FastAPI app setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "message_text": ""
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, message: str = Form(...)):
    transformed_sms = transform_text(message)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    prediction = "⚠️ Spam Message Detected!" if result == 1 else "✅ This is a Ham (Not Spam) message."

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": prediction,
        "message_text": message
    })

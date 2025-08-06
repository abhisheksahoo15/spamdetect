from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import string
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Setup
ps = PorterStemmer()
templates = Jinja2Templates(directory="templates")
app = FastAPI()

# Manually defined stopwords (to avoid nltk stopwords dependency)
stopwords_set = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
    "won't", 'wouldn', "wouldn't"
}

def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenization (basic)
    words = []
    for word in text.split():
        word = word.strip(string.punctuation)
        if word and word.isalnum():
            words.append(word)

    # Remove stopwords
    words = [word for word in words if word not in stopwords_set]

    # Stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)

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

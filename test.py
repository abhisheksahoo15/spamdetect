import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open("modeletc3.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

# Sample test data
test_samples = [
    "Congratulations! You've won a free ticket to Bahamas. Reply WIN to claim now.",
    "Hey, are you coming to the class tomorrow?",
    "FREE entry in 2 a weekly competition to win FA Cup final tickets!",
    "Please submit the report by tonight.",
    "You’ve been selected for a cash prize. Call now to receive."
]

# Test model
for text in test_samples:
    transformed = transform_text(text)
    vector_input = vectorizer.transform([transformed])  # Fix: convert to vector
    prediction = model.predict(vector_input)[0]
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    print(f"Input: {text}\nPrediction: {label}\n{'-'*50}")

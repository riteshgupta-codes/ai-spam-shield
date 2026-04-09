import streamlit as st
import pickle
import string
import re
import nltk
import time
import requests
from streamlit_lottie import st_lottie
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load NLTK
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Spam Shield", page_icon="🛡️", layout="centered")

# --- CUSTOM CSS FOR DESIGN ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTextArea textarea { border-radius: 15px; border: 2px solid #dfe1e5; }
    .stButton button {
        width: 100%; border-radius: 25px; height: 3em;
        background-color: #4CAF50; color: white; font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover { background-color: #45a049; transform: scale(1.02); }
    .result-box {
        padding: 20px; border-radius: 15px; text-align: center;
        font-size: 24px; font-weight: bold; margin-top: 20px;
        animation: fadeIn 1.5s;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie Animations with Error Handling
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Updated to a new, stable Lottie URL
lottie_url = "https://lottie.host/74737750-f8f4-419b-b236-09e44d32626e/8O5E9r0Gst.json"
lottie_processing = load_lottieurl(lottie_url)

# --- LOGIC ---
def transform_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    tokens = nltk.word_tokenize(text)
    stop_words = stopwords.words('english')
    y = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop_words and w not in string.punctuation]
    return " ".join(y)

# Load Models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- UI BODY ---
st.title("🛡️ AI Spam Shield")
st.write("Ensuring your inbox stays clean with advanced machine learning.")

message = st.text_area("Drop your message here...", placeholder="Type or paste a message to analyze...")

if st.button('Verify Message'):
    if message.strip() == "":
        st.warning("Please enter a message first!")
    else:
        with st.spinner('Analyzing patterns...'):
            # SAFETY CHECK: Only show animation if it loaded correctly
            if lottie_processing:
                st_lottie(lottie_processing, height=150, key="loader")
            
            time.sleep(1.5) 
            
            transformed_msg = transform_text(message)
            vector_input = tfidf.transform([transformed_msg])
            result = model.predict(vector_input)[0]

        # Step 2: Styled Display
        if result == 1:
            st.markdown('<div class="result-box" style="background-color: #ffebee; color: #c62828; border: 2px solid #ef9a9a;">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
            st.snow() 
        else:
            st.markdown('<div class="result-box" style="background-color: #e8f5e9; color: #2e7d32; border: 2px solid #a5d6a7;">✅ NOT SPAM</div>', unsafe_allow_html=True)
            st.balloons()
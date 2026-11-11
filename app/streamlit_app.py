
import os
import re
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

st.set_page_config(page_title="Smart Fake News Detection", page_icon="‘’")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("‘’ Smart Fake News Detection System")
st.write("Enter a news article below to classify its text.")

news_text = st.text_area("Enter news text here:", height=200)

if st.button("Check News"):
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        try:
            model, vectorizer = load_artifacts()
            cleaned_text = clean_text(news_text)
            transformed_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(transformed_text)[0]

            if prediction == 0:
                st.error("Prediction: Fake News")
            else:
                st.success("Prediction: Real News")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_text)[0]
                confidence = max(probabilities) * 100
                st.write(f"Model confidence estimate: {confidence:.2f}%")

        except Exception as error:
            st.error("Unable to make a prediction.")
            st.write(str(error))

st.info(
    "Disclaimer: This is an educational machine-learning project. "
    "Its predictions are not proof that a news article is true or false. "
    "Please verify news using reliable sources."
)

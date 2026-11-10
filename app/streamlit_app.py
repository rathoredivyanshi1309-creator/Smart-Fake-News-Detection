
import streamlit as st
import joblib
import re
import os

# Locate saved model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.joblib")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

# Load trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Clean input text
def clean_text(text):
    text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"[^a-z\s]", " ", text)
                text = re.sub(r"\s+", " ", text)
                    return text.strip()

                    # Application title
                    st.set_page_config(
                        page_title="Smart Fake News Detection",
                            page_icon="📰",
                                layout="centered"
                                )

                                st.title("📰 Smart Fake News Detection System")

                                st.write(
                                    "Enter a news headline or article below "
                                        "to see the model's prediction."
                                        )

                                        # News input
                                        news_input = st.text_area(
                                            "Enter news text:",
                                                height=200,
                                                    placeholder="Paste your news headline or article here..."
                                                    )

                                                    # Prediction button
                                                    if st.button("Predict News"):

                                                        if not news_input.strip():
                                                                st.warning("Please enter some news text first.")

                                                                    else:
                                                                            cleaned_input = clean_text(news_input)

                                                                                    if not cleaned_input:
                                                                                                st.warning("Please enter meaningful news text.")

                                                                                                        else:
                                                                                                                    input_vector = vectorizer.transform([cleaned_input])

                                                                                                                                prediction = model.predict(input_vector)[0]

                                                                                                                                            probabilities = model.predict_proba(input_vector)[0]
                                                                                                                                                        confidence = max(probabilities) * 100

                                                                                                                                                                    st.subheader("Prediction Result")

                                                                                                                                                                                if prediction == 0:
                                                                                                                                                                                                st.error("The model classifies this news as FAKE.")
                                                                                                                                                                                                            else:
                                                                                                                                                                                                                            st.success("The model classifies this news as REAL.")

                                                                                                                                                                                                                                        st.write(
                                                                                                                                                                                                                                                        f"Model confidence score: {confidence:.2f}%"
                                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                                                st.progress(float(confidence / 100))

                                                                                                                                                                                                                                                                                            st.info(
                                                                                                                                                                                                                                                                                                            "This confidence score reflects the model's "
                                                                                                                                                                                                                                                                                                                            "prediction, not the verified truth of the news."
                                                                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                                                                        st.divider()

                                                                                                                                                                                                                                                                                                                                        st.subheader("Important Disclaimer")

                                                                                                                                                                                                                                                                                                                                        st.write(
                                                                                                                                                                                                                                                                                                                                            "This application is an educational machine-learning project. "
                                                                                                                                                                                                                                                                                                                                                "It does not independently verify news claims or consult "
                                                                                                                                                                                                                                                                                                                                                    "live fact-checking sources. Predictions may be incorrect, "
                                                                                                                                                                                                                                                                                                                                                        "especially for recent events or news outside the training "
                                                                                                                                                                                                                                                                                                                                                            "dataset. Always verify important information using "
                                                                                                                                                                                                                                                                                                                                                                "reliable sources."
                                                                                                                                                                                                                                                                                                                                                                )
                                                                                                                                                                                                                                                                                                                                        


# Smart Fake News Detection System

## About the Project
The Smart Fake News Detection System is a machine
learning project that classifies news articles as
Fake News or Real News using Natural Language
Processing (NLP).

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Joblib

## Dataset
The project uses the ISOT Fake News Dataset,
containing labelled fake and real news articles.

## Data Processing
- Missing value handling
- Duplicate removal
- Text cleaning
- Lowercase conversion
- TF-IDF feature extraction
- Training and testing split

## Machine Learning Models
1. Logistic Regression
2. Multinomial Naive Bayes

## Model Performance

| Metric | Logistic Regression | Naive Bayes |
|---|---:|---:|
| Accuracy | 98.43% | 93.04% |
| Precision | 97.86% | 93.76% |
| Recall | 99.32% | 93.54% |
| F1 Score | 98.58% | 93.65% |

Logistic Regression was selected based on its
higher F1-score.

## Application Features
- News headline or article input
- Fake or Real prediction
- Model confidence score
- Input validation
- Disclaimer for users

## How to Run
Install the required packages:

pip install -r requirements.txt

Run the application:

streamlit run app/streamlit_app.py

## Project Limitations
This application does not independently verify
news claims or access live fact-checking sources.
Its predictions may be incorrect, particularly
for recent events or unfamiliar topics.

Always verify important news using reliable
sources.

## Author
Smart Fake News Detection Project

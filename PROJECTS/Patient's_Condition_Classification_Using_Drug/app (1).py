import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------
# Load saved objects
# -----------------------
model = pickle.load(open("condition_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
drug_stats = pickle.load(open("drug_stats.pkl", "rb"))

# -----------------------
# Text cleaning
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text

# -----------------------
# Drug recommendation
# -----------------------
def recommend_best_drug(condition, top_n=3):
    filtered = drug_stats[drug_stats['condition'] == condition]

    if filtered.empty:
        return None

    return (
        filtered
        .sort_values(by='bayesian_score', ascending=False)
        .head(top_n)
    )

# -----------------------
# Prediction pipeline
# -----------------------
def predict_condition_and_recommend_drug(review_text):
    cleaned = clean_text(review_text)
    vec = tfidf.transform([cleaned])

    label = model.predict(vec)[0]
    condition = le.inverse_transform([label])[0]

    drugs = recommend_best_drug(condition)
    return condition, drugs

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Medical NLP Drug Recommendation", layout="centered")

st.title("💊 Medical Condition Prediction & Drug Recommendation")
st.write("Predicts medical condition from patient reviews and recommends the best drug using Bayesian scoring.")

review = st.text_area(
    "Enter patient review:",
    placeholder="Example: I have severe stomach pain and acid reflux for weeks..."
)

if st.button("Predict & Recommend"):
    if review.strip() == "":
        st.warning("Please enter a medical review.")
    else:
        condition, drugs = predict_condition_and_recommend_drug(review)

        st.success("Prediction Successful")

        st.subheader("🩺 Predicted Condition")
        st.write(condition)

        st.subheader("💊 Recommended Drugs")

        if drugs is None:
            st.info("No drug data available for this condition.")
        else:
            st.dataframe(
                drugs[['drugName', 'avg_rating', 'review_count', 'bayesian_score']]
                .rename(columns={
                    'drugName': 'Drug Name',
                    'avg_rating': 'Avg Rating',
                    'review_count': 'Reviews',
                    'bayesian_score': 'Bayesian Score'
                })
            )

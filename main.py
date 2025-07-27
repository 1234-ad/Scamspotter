
import streamlit as st
from src.logic import predict_message, train_model
import os

st.set_page_config(page_title='ScamSpotter', layout='centered')

st.title("🔍 ScamSpotter - Scam Message Detector")

if not os.path.exists('src/model.pkl'):
    st.write("Training model for the first time...")
    train_model()

user_input = st.text_area("Enter a message to check if it's a scam or safe:")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        label, confidence = predict_message(user_input)
        st.markdown(f"### Result: **{label}** (Confidence: {confidence*100:.2f}%)")

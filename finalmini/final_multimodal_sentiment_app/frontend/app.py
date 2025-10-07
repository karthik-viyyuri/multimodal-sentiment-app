
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Multimodal Sentiment App", layout="centered", page_icon="💬")
st.title("🧠 Multimodal Sentiment Classifier")
st.markdown("Upload an image or type a sentence to predict sentiment (positive, neutral, negative).")

tab1, tab2 = st.tabs(["📄 Text", "🖼️ Image"])

with tab1:
    user_text = st.text_area("Enter a sentence:")
    if st.button("Analyze Text"):
        if user_text.strip():
            response = requests.post("http://localhost:8000/analyze-text", json={"text": user_text})
            if response.status_code == 200:
                result = response.json()
                label = ["Negative", "Neutral", "Positive"][result["sentiment"] + 1]
                st.success(f"**Sentiment:** {label} (via {result['source']})")
            else:
                st.error("Failed to connect to backend.")

with tab2:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=300)
    if st.button("Analyze Image"):
        if uploaded_file:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://localhost:8000/analyze-image", files={"file": uploaded_file})
            if response.status_code == 200:
                result = response.json()
                label = ["Negative", "Neutral", "Positive"][result["sentiment"] + 1]
                st.success(f"**Sentiment:** {label} (via {result['source']})")
            else:
                st.error("Failed to connect to backend.")

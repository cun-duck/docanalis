import streamlit as st
from transformers import pipeline
from huggingface_hub import login
import os
import docx2txt
from PyPDF2 import PdfReader

# Login with Hugging Face token stored in Streamlit Secrets
hf_token = st.secrets["huggingface"]["token"]

# Initialize models for different tasks
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
keyword_extractor = pipeline("feature-extraction", model="distilbert-base-uncased")

# Title of the application
st.title("Document Analysis Application")

# File uploader
uploaded_file = st.file_uploader("Upload your document (PDF, Word, or Text)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Initialize an empty text variable
    text = ""

    # Read the content of the uploaded file
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle None if no text is found
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(uploaded_file)  # Use docx2txt for .docx files

    # Check if text is empty after processing
    if not text.strip():
        st.warning("The document is empty or contains no readable text.")
    else:
        # Display document preview
        st.subheader("Document Preview")
        st.text(text[:1000] + "...")  # Show only the first 1000 characters

        # Summarization
        st.subheader("Summarization")
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
        st.write(summary[0]["summary_text"])

        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        sentiment = sentiment_analyzer(text)
        st.write(f"Sentiment: {sentiment[0]['label']} with confidence {sentiment[0]['score']:.2f}")

        # Named Entity Recognition (NER)
        st.subheader("Named Entity Recognition (NER)")
        entities = ner(text)
        entities_df = [{"Entity": entity["word"], "Label": entity["entity"]} for entity in entities]
        st.write(entities_df)

        # Keyword Extraction (using sentence-transformers model for feature extraction)
        st.subheader("Keyword Extraction")
        features = keyword_extractor(text)
        # Extract the first layer (embedding for each word) for simplicity
        keywords = [word for word, _ in zip(text.split(), features[0])]
        st.write(", ".join(keywords[:10]))  # Display top 10 keywords

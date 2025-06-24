import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Keyword Category Classifier", layout="wide")
st.title("🎯 Keyword Category Classifier")

st.markdown("""
Paste your keywords below. This tool will automatically assign each keyword to a specific product/topic category using NLP.
""")

# Keyword input area
keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

# Pre-trained model setup (DistilBERT for text embeddings)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Available categories for classification
categories = [
    "Photo Editor", "Best Photo Editor", "Best Photo Editor Software", "Photo Editor Software"
]

# Function to compute the embeddings of a text using BERT-like model
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Function to classify keywords and map intent
def classify_keywords(keywords):
    category_labels = []
    intent_labels = []
    progress = st.progress(0)
    status_text = st.empty()

    # Compute embeddings for categories
    category_embeddings = {category: get_embeddings(category) for category in categories}

    try:
        for i, kw in enumerate(keywords):
            if not kw.strip():
                continue  # Skip empty keywords

            # Compute the embedding of the keyword
            kw_embedding = get_embeddings(kw)

            # Calculate cosine similarity between keyword and each category
            similarities = {category: cosine_similarity(kw_embedding.unsqueeze(0), embedding.unsqueeze(0))[0][0]
                            for category, embedding in category_embeddings.items()}
            
            # Assign the category with the highest similarity
            assigned_category = max(similarities, key=similarities.get)
            category_labels.append(assigned_category)

            # Intent logic based on keyword patterns
            intent = "Generic"  # Default intent

            if "free" in kw.lower() and "online" in kw.lower():
                intent = "Free & Online"
            elif "free" in kw.lower():
                intent = "Free"
            elif "online" in kw.lower():
                intent = "Online"
            elif "software" in kw.lower():
                intent = "Software"
            elif "certification" in kw.lower():
                intent = "Certification"
            
            intent_labels.append(intent)

            progress.progress((i + 1) / len(keywords))
            status_text.text(f"Classified {i + 1} of {len(keywords)}")
            time.sleep(0.5)

        return category_labels, intent_labels

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return [], []

# Button to trigger the classification
if keyword_input.strip():
    if st.button("Classify Keywords"):
        keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]
        df = pd.DataFrame({"Keyword": keywords})

        # Classify the keywords
        category_labels, intent_labels = classify_keywords(keywords)

        if category_labels:
            df["Assigned Category"] = category_labels
            df["Intent"] = intent_labels
            st.success("Classification complete!")
            st.dataframe(df)

            # Provide download button for CSV
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Classified CSV", data=csv, file_name="classified_keywords.csv", mime="text/csv")

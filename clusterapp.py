import streamlit as st
import pandas as pd
from transformers import pipeline
import torch
import time

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Keyword Category Classifier", layout="wide")
st.title("🎯 Keyword Category Classifier")

st.markdown("""
Paste your keywords below. This tool will automatically assign each keyword to a specific product/topic category using a pre-trained model.
""")

# Keyword input area
keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

# Dynamically set the device to GPU if available, else use CPU
device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU

# Initialize the zero-shot classifier pipeline
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased", device=device)

# Available categories for classification
candidate_labels = [
    "Pricing", "Certification", "Course", "Education", "Financials", 
    "Branding", "Marketing", "Technology", "Cloud", "Business"
]

# Function to classify keywords and map intent
def classify_keywords(keywords):
    category_labels = []
    intent_labels = []
    progress = st.progress(0)
    status_text = st.empty()

    try:
        for i, kw in enumerate(keywords):
            if not kw.strip():
                continue  # Skip empty keywords

            # Predict the category using zero-shot classification
            result = classifier(kw, candidate_labels)
            category = result['labels'][0]  # Take the top predicted label
            category_labels.append(category)

            # Intent logic based on keyword patterns
            intent = "Generic"  # Default intent

            if "free" in kw.lower() and "online" in kw.lower():
                intent = "Free & Online"
            elif "free" in kw.lower():
                intent = "Free"
            elif "online" in kw.lower():
                intent = "Online"
            elif "course" in kw.lower() or "tutorial" in kw.lower():
                intent = "Course"
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

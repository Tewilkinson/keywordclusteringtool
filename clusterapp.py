import streamlit as st
import pandas as pd
from transformers import pipeline
import time
import torch

# --- Streamlit UI ---
st.set_page_config(page_title="Keyword Classifier", layout="wide")
st.title("🎯 Keyword Category Classifier")

st.markdown("""
Paste your keywords below. This tool will automatically assign each keyword to a specific product/topic category using a pre-trained model.
""")

# Keyword input area
keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

# Ensure the button is only shown when there is some input
if keyword_input.strip():
    # Use a smaller model for better performance if resources are limited
    classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")

    # Available categories for classification
    candidate_labels = [
        "Pricing", "Certification", "Course", "Education", "Financials", 
        "Branding", "Marketing", "Technology", "Cloud", "Business"
    ]

    # Function to classify keywords
    def classify_keywords(keywords):
        category_labels = []
        progress = st.progress(0)
        status_text = st.empty()

        try:
            for i, kw in enumerate(keywords):
                if not kw.strip():
                    continue  # Skip empty keywords
                
                # Use the zero-shot classification to predict categories for each keyword
                result = classifier(kw, candidate_labels)
                category = result['labels'][0]  # Take the top predicted label
                category_labels.append(category)
                
                progress.progress((i + 1) / len(keywords))
                status_text.text(f"Classified {i + 1} of {len(keywords)}")
                time.sleep(0.5)

            return category_labels

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return []

    # Button for classifying the keywords
    if st.button("Classify Keywords"):
        keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]
        df = pd.DataFrame({"Keyword": keywords})

        # Classify keywords
        category_labels = classify_keywords(keywords)

        if category_labels:
            df["Assigned Category"] = category_labels
            st.success("Classification complete!")
            st.dataframe(df)

            # Provide download button for CSV
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Classified CSV", data=csv, file_name="classified_keywords.csv", mime="text/csv")

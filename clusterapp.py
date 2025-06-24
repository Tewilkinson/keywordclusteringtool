import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import time

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Keyword Category Classifier", layout="wide")
st.title("🎯 Keyword Category Classifier")

st.markdown("""
Paste your keywords below. This tool will automatically categorize the keywords based on their meaning using NLP.
""")

# Keyword input area
keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

# Initialize the Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to classify keywords based on embeddings and clustering
def classify_keywords(keywords):
    category_labels = []
    intent_labels = []
    progress = st.progress(0)
    status_text = st.empty()

    try:
        # Encode the keywords into embeddings
        embeddings = model.encode(keywords)

        # Use KMeans clustering to group similar keywords
        num_clusters = 3  # You can change the number of clusters based on your dataset
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(embeddings)

        # Assign categories based on the cluster centers
        cluster_centers = kmeans.cluster_centers_
        for i, kw in enumerate(keywords):
            cluster = kmeans.predict([embeddings[i]])[0]
            cluster_center = cluster_centers[cluster]

            # Category assignment based on the closest cluster center
            if np.dot(cluster_center, embeddings[i]) > 0.8:
                category_labels.append(f"Cluster {cluster + 1}")
            else:
                category_labels.append("Unclassified")

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
            elif "course" in kw.lower() or "tutorial" in kw.lower():
                intent = "Course"
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

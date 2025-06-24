import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Keyword Category Classifier", layout="wide")
st.title("🎯 Keyword Category Classifier")

st.markdown("""
Paste your keywords below. This tool will automatically categorize the keywords into clusters and name each cluster based on the keywords it contains.
""")

# Keyword input area
keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

# Initialize the Sentence-BERT model for generating embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to classify keywords based on embeddings and clustering
def classify_keywords(keywords):
    category_labels = []
    progress = st.progress(0)
    status_text = st.empty()

    try:
        # Encode the keywords into embeddings
        embeddings = model.encode(keywords)

        # Calculate the pairwise cosine similarity between keywords
        similarity_matrix = cosine_similarity(embeddings)

        # Use the cosine similarity to cluster keywords
        # A very basic clustering: keywords with high similarity get grouped together
        threshold = 0.5  # Similarity threshold to consider keywords as belonging to the same cluster
        cluster_labels = [-1] * len(keywords)  # Start with all keywords unclassified
        cluster_count = 0
        
        for i in range(len(keywords)):
            if cluster_labels[i] == -1:  # If not yet classified
                cluster_labels[i] = cluster_count
                # Find all keywords that have high similarity to this one
                for j in range(i + 1, len(keywords)):
                    if cluster_labels[j] == -1 and similarity_matrix[i][j] > threshold:
                        cluster_labels[j] = cluster_count
                cluster_count += 1

        # Assign categories based on cluster assignment
        cluster_keywords = {i: [] for i in range(cluster_count)}  # Initialize empty lists for each cluster
        for idx, label in enumerate(cluster_labels):
            cluster_keywords[label].append(keywords[idx])

        # Assign cluster names based on the most representative keywords
        cluster_names = []
        for cluster in cluster_keywords:
            # Use the most frequent keyword in the cluster as the cluster name
            cluster_keywords_list = cluster_keywords[cluster]
            cluster_name = max(set(cluster_keywords_list), key=cluster_keywords_list.count)
            cluster_names.append(cluster_name)

            # Assign the cluster name to the respective keywords
            for kw in cluster_keywords_list:
                category_labels.append(cluster_name)

            progress.progress((cluster + 1) / cluster_count)
            status_text.text(f"Cluster {cluster + 1} processed")
            time.sleep(0.5)

        # Create the DataFrame with category labels
        df = pd.DataFrame({"Keyword": keywords, "Assigned Category": category_labels})

        return df, cluster_names

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return pd.DataFrame(), []

# Button to trigger the classification
if keyword_input.strip():
    if st.button("Classify Keywords"):
        keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]
        df, cluster_names = classify_keywords(keywords)

        if not df.empty:
            # Display the dataframe with assigned categories
            st.success("Classification complete!")
            st.dataframe(df)

            # Display the cluster names (representative names based on clusters)
            st.write("Cluster Names based on Keywords:")
            for i, cluster_name in enumerate(cluster_names, 1):
                st.write(f"Cluster {i}: {cluster_name}")

            # Provide download button for CSV
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Classified CSV", data=csv, file_name="classified_keywords.csv", mime="text/csv")

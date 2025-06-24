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

        # Use KMeans clustering to group similar keywords
        num_clusters = 3  # Adjust number of clusters based on data
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(embeddings)

        # Assign categories and cluster names based on the cluster centers
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        # Define possible cluster names based on known keywords
        photo_keywords = ["photo", "editing", "editor", "photo editor", "image"]
        video_keywords = ["video", "editing", "editor", "video editor", "film"]
        
        cluster_names = []
        for cluster in range(num_clusters):
            # Get the indices of keywords in the current cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]

            # Extract the keywords that belong to this cluster
            cluster_keywords = [keywords[i] for i in cluster_indices]

            # Determine if the cluster is related to photo or video based on keyword occurrence
            photo_count = sum(any(word in kw.lower() for word in photo_keywords) for kw in cluster_keywords)
            video_count = sum(any(word in kw.lower() for word in video_keywords) for kw in cluster_keywords)

            # Name the cluster based on the most frequent category
            if photo_count > video_count:
                cluster_name = "Photo Editor"
            elif video_count > photo_count:
                cluster_name = "Video Editor"
            else:
                cluster_name = "General"

            cluster_names.append(cluster_name)

            # Assign the cluster name to the respective keywords
            for kw in cluster_keywords:
                category_labels.append(cluster_name)

            progress.progress((cluster + 1) / num_clusters)
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

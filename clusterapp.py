import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
import numpy as np
import re

def main():
    st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Auto-Clustering Keyword Tool")
    st.markdown("""
    Paste your keywords (one per line) and let BERT + Affinity Propagation  
    automatically discover clusters and name them by a core phrase.
    """)

    keyword_input = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        # 1. Embed + normalize
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(keywords, normalize_embeddings=True)

        # 2. Build cosine-similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)

        # 3. Affinity Propagation (auto-clusters + exemplars)
        ap = AffinityPropagation(affinity="precomputed", random_state=42)
        ap.fit(sim_matrix)
        labels = ap.labels_

        # 4. Cluster naming by core token + shortest phrase
        stop_words = {"best", "free", "online", "software", "generator"}
        cluster_names = {}

        for cluster_id in np.unique(labels):
            members = np.where(labels == cluster_id)[0]

            # collect non-stop tokens
            tokens = []
            for idx in members:
                for w in re.findall(r"\w+", keywords[idx].lower()):
                    if w not in stop_words:
                        tokens.append(w)

            primary = max(set(tokens), key=tokens.count) if tokens else ""
            candidates = [keywords[idx] for idx in members if primary in keywords[idx].lower()]
            if candidates:
                name = min(candidates, key=lambda s: len(s))
            else:
                name = keywords[members[0]]

            cluster_names[cluster_id] = name

        # 5. Build results DataFrame
        df = pd.DataFrame({
            "Keyword": keywords,
            "Cluster": [cluster_names[label] for label in labels]
        })

        # — Display —
        st.success(f"Found {len(cluster_names)} clusters.")
        st.dataframe(df)

        # — Download —
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name="clusters.csv", mime="text/csv")

if __name__ == "__main__":
    main()

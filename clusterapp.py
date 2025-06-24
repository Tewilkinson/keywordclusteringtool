import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
import numpy as np

st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
st.title("🤖 Auto-Clustering Keyword Tool")

st.markdown("""
Paste your keywords (one per line) and let BERT + Affinity Propagation  
automatically discover clusters and name them by exemplar queries.
""")

keywords = [k.strip() for k in st.text_area("🔤 Keywords:", height=300).splitlines() if k.strip()]

if st.button("Cluster Keywords") and keywords:
    # 1. Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(keywords, normalize_embeddings=True)

    # 2. Cosine-similarity matrix
    sim_matrix = np.dot(embeddings, embeddings.T)

    # 3. Affinity Propagation (auto-n_clusters + exemplars)
    ap = AffinityPropagation(affinity="precomputed", random_state=42)
    ap.fit(sim_matrix)
    labels = ap.labels_
    exemplars = ap.cluster_centers_indices_

    # 4. Build cluster names from exemplars
    cluster_names = {i: keywords[idx] for i, idx in enumerate(exemplars)}

    # 5. Assemble DataFrame
    df = pd.DataFrame({
        "Keyword": keywords,
        "Cluster": [ cluster_names[label] for label in labels ]
    })

    st.success(f"Found {len(cluster_names)} clusters.")
    st.dataframe(df)

    # Show legend of cluster → exemplar
    st.write("### Cluster exemplars (names):")
    for i, name in cluster_names.items():
        st.write(f"• **Cluster {i}** → {name}")

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "clusters.csv", "text/csv")

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
import numpy as np

st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
st.title("🤖 Auto-Clustering Keyword Tool")

st.markdown("""
Paste your keywords (one per line) and let BERT + Affinity Propagation  
automatically discover clusters and name them by their most central keyword.
""")

keywords = [k.strip() for k in st.text_area("🔤 Keywords:", height=300).splitlines() if k.strip()]

if st.button("Cluster Keywords") and keywords:
    # 1. Embed + normalize
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(keywords, normalize_embeddings=True)

    # 2. Cosine-similarity matrix
    sim = np.dot(emb, emb.T)

    # 3. Affinity Propagation
    ap = AffinityPropagation(affinity="precomputed", random_state=42)
    ap.fit(sim)
    labels = ap.labels_

    # 4. For each cluster, pick the medoid (member with highest total similarity) as name
    cluster_names = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        # sum of similarities within cluster
        intra_sim = sim[np.ix_(members, members)].sum(axis=1)
        medoid = members[np.argmax(intra_sim)]
        cluster_names[c] = keywords[medoid]

    # 5. Build DataFrame
    df = pd.DataFrame({
        "Keyword": keywords,
        "Cluster": [cluster_names[l] for l in labels]
    })

    st.success(f"Found {len(cluster_names)} clusters.")
    st.dataframe(df)

    st.write("### Cluster names (medoids):")
    for c, name in cluster_names.items():
        st.write(f"• Cluster {c} → {name}")

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "clusters.csv", "text/csv")

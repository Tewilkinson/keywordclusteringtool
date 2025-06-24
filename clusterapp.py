import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# — Cache the model —
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def cluster_and_name(keywords, model, progress, status_text):
    total_steps = 5
    step = 1

    # 1) Embed
    status_text.text(f"{step}/{total_steps} – Embedding keywords…")
    emb = model.encode(keywords, normalize_embeddings=True)
    progress.progress(step / total_steps)
    step += 1

    # 2) Build similarity matrix
    status_text.text(f"{step}/{total_steps} – Building similarity matrix…")
    sim = np.dot(emb, emb.T)
    progress.progress(step / total_steps)
    step += 1

    # 3) Threshold-based clustering via union-find
    status_text.text(f"{step}/{total_steps} – Clustering (threshold)…")
    N = len(keywords)
    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    threshold = 0.6  # tweak up/dn for tighter/looser clusters
    for i in range(N):
        for j in range(i + 1, N):
            if sim[i][j] > threshold:
                union(i, j)

    # Build clusters
    clusters = {}
    for i in range(N):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    progress.progress(step / total_steps)
    step += 1

    # 4) Name clusters by core token + shortest phrase
    status_text.text(f"{step}/{total_steps} – Naming clusters…")
    stop_words = {"best", "free", "online", "software", "generator"}
    cluster_names = {}
    for root, inds in clusters.items():
        tokens = []
        for i in inds:
            for w in re.findall(r"\w+", keywords[i].lower()):
                if w not in stop_words:
                    tokens.append(w)
        if tokens:
            primary = max(set(tokens), key=tokens.count)
            cands = [keywords[i] for i in inds if primary in keywords[i].lower()]
            name = min(cands, key=len) if cands else keywords[inds[0]]
        else:
            name = keywords[inds[0]]
        cluster_names[root] = name

    progress.progress(step / total_steps)
    step += 1

    # 5) Assemble DataFrame
    status_text.text("Done!")
    df = pd.DataFrame({
        "Keyword": keywords,
        "Cluster": [cluster_names[find(i)] for i in range(N)]
    })
    progress.progress(1.0)
    return df

def main():
    st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Auto-Clustering Keyword Tool")
    st.markdown("""
    Paste your keywords (one per line).  
    This runs a fast similarity-threshold clustering on BERT embeddings,  
    merges similar queries, and names each cluster by its core phrase.
    """)

    raw = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        model = load_model()

        progress = st.progress(0.0)
        status_text = st.empty()

        try:
            df = cluster_and_name(keywords, model, progress, status_text)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="clusters.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

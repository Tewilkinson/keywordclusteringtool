import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def main():
    st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Auto-Clustering Keyword Tool")
    st.markdown("""
    Paste your keywords (one per line) and let BERT + Affinity Propagation  
    automatically discover & name clusters (singletons get merged).
    """)

    # 1) Input
    raw = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        # 2) Embed + normalize
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(keywords, normalize_embeddings=True)

        # 3) Cosine-similarity matrix
        sim = np.dot(emb, emb.T)

        # 4) Affinity Propagation
        ap = AffinityPropagation(affinity="precomputed", random_state=42)
        ap.fit(sim)
        labels = ap.labels_.copy()

        # 5) Build initial clusters
        clusters = {
            cid: [i for i, lab in enumerate(labels) if lab == cid]
            for cid in np.unique(labels)
        }

        # 6) Merge any singleton clusters into nearest cluster
        #    Compute a mean-embedding for each cluster
        cluster_means = {
            cid: emb[inds].mean(axis=0)
            for cid, inds in clusters.items()
        }
        for cid, inds in list(clusters.items()):
            if len(inds) == 1:
                idx = inds[0]
                # find best other cluster by similarity to its mean
                sims = {
                    other: cosine_similarity(
                        emb[idx:idx+1], mean.reshape(1, -1)
                    )[0][0]
                    for other, mean in cluster_means.items()
                    if other != cid
                }
                best = max(sims

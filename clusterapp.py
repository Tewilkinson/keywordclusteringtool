import streamlit as st
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# — Cache the model —
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def name_cluster(keywords, inds):
    stop = {"best","free","online","software","generator"}
    tokens = []
    for i in inds:
        for w in re.findall(r"\w+", keywords[i].lower()):
            if w not in stop:
                tokens.append(w)
    if tokens:
        core = max(set(tokens), key=tokens.count)
        cands = [keywords[i] for i in inds if core in keywords[i].lower()]
        return min(cands, key=len) if cands else keywords[inds[0]]
    return keywords[inds[0]]

def cluster_hierarchy(keywords, model, progress, status):
    steps = 6
    step = 1

    # 1. Embed
    status.text(f"{step}/{steps} – Embedding…")
    emb = model.encode(keywords, normalize_embeddings=True)
    progress.progress(step/steps); step += 1

    # 2. Cosine distance
    status.text(f"{step}/{steps} – Computing distances…")
    sim = cosine_similarity(emb)
    dist = 1 - sim
    progress.progress(step/steps); step += 1

    # 3. Top‐level clustering
    status.text(f"{step}/{steps} – Top‐level clustering…")
    top_cluster = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=0.6
    )
    top_labels = top_cluster.fit_predict(dist)
    progress.progress(step/steps); step += 1

    # 4. Name top clusters
    status.text(f"{step}/{steps} – Naming top clusters…")
    top_clusters = {}
    for i, lab in enumerate(top_labels):
        top_clusters.setdefault(lab, []).append(i)
    top_names = {lab: name_cluster(keywords, inds)
                 for lab, inds in top_clusters.items()}
    progress.progress(step/steps); step += 1

    # 5. Sub‐clustering within each top cluster
    status.text(f"{step}/{steps} – Sub‐clustering…")
    rows = []
    for lab, inds in top_clusters.items():
        if len(inds) == 1:
            sub_labels = [0]
            sub_names = {0: keywords[inds[0]]}
        else:
            sub_dist = dist[np.ix_(inds, inds)]
            sub_cluster = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=0.3
            )
            local_labels = sub_cluster.fit_predict(sub_dist)
            grouped = {}
            for idx, sl in zip(inds, local_labels):
                grouped.setdefault(sl, []).append(idx)
            sub_names = {
                sl: name_cluster(keywords, sub_inds)
                for sl, sub_inds in grouped.items()
            }
            sub_labels = local_labels

        for idx, sl in zip(inds, sub_labels):
            rows.append({
                "Keyword": keywords[idx],
                "Cluster": top_names[lab],
                "Subcluster": sub_names[sl]
            })
    progress.progress(step/steps); step += 1

    # 6. Done
    status.text("Done!")
    progress.progress(1.0)
    return pd.DataFrame(rows)

def main():
    st.set_page_config(page_title="Hierarchical Clustering", layout="wide")
    st.title("🔗 Hierarchical Auto-Clustering Tool")
    st.markdown("""
1. Level 1: Agglomerative (cosine-distance, threshold=0.6) → top clusters  
2. Level 2: Agglomerative within each top cluster (threshold=0.3) → subclusters  
3. Names by core token + shortest phrase; fully automatic.  
Tweak `distance_threshold` for finer/coarser grouping.
    """)

    raw = st.text_area("Enter keywords (one per line):", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        model = load_model()
        progress = st.progress(0.0)
        status = st.empty()
        try:
            df = cluster_hierarchy(keywords, model, progress, status)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv,
                               file_name="clusters.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__=="__main__":
    main()

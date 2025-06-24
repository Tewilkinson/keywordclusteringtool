import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# — cache the embedding model —
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def union_find(similarity, threshold):
    """
    Given an NxN sim matrix, cluster via union-find on sim>threshold.
    Returns a list 'labels' of length N.
    """
    N = similarity.shape[0]
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

    for i in range(N):
        for j in range(i+1, N):
            if similarity[i, j] > threshold:
                union(i, j)

    # finalize labels
    labels = [find(i) for i in range(N)]
    # reindex to 0…K-1
    uniq = sorted(set(labels))
    mapping = {old: new for new, old in enumerate(uniq)}
    return [mapping[l] for l in labels]

def name_cluster(keywords, inds, stop_words):
    # pick core token + shortest phrase
    tokens = []
    for i in inds:
        for w in re.findall(r"\w+", keywords[i].lower()):
            if w not in stop_words:
                tokens.append(w)
    if tokens:
        primary = max(set(tokens), key=tokens.count)
        cands = [keywords[i] for i in inds if primary in keywords[i].lower()]
        return min(cands, key=len) if cands else keywords[inds[0]]
    else:
        return keywords[inds[0]]

def cluster_and_hierarchy(keywords, model, progress, status_text):
    steps = 7
    step = 1

    # 1) Embed
    status_text.text(f"{step}/{steps} – Embedding keywords…")
    emb = model.encode(keywords, normalize_embeddings=True)
    progress.progress(step/steps)
    step += 1

    # 2) Similarity matrix
    status_text.text(f"{step}/{steps} – Building similarity matrix…")
    sim = np.dot(emb, emb.T)
    progress.progress(step/steps)
    step += 1

    # 3) Top‐level clustering
    status_text.text(f"{step}/{steps} – Top‐level clustering…")
    top_labels = union_find(sim, threshold=0.5)
    progress.progress(step/steps)
    step += 1

    # 4) Name top clusters
    status_text.text(f"{step}/{steps} – Naming top clusters…")
    stop_words = {"best","free","online","software","generator"}
    top_clusters = {}
    for idx, lab in enumerate(top_labels):
        top_clusters.setdefault(lab, []).append(idx)
    top_names = {
        lab: name_cluster(keywords, inds, stop_words)
        for lab, inds in top_clusters.items()
    }
    progress.progress(step/steps)
    step += 1

    # 5) Sub-clustering within each top cluster
    status_text.text(f"{step}/{steps} – Sub-clustering…")
    sub_labels = [None]*len(keywords)
    sub_names = {}
    for lab, inds in top_clusters.items():
        if len(inds)==1:
            # single => stays alone
            sub_labels[inds[0]] = 0
            sub_names[(lab,0)] = keywords[inds[0]]
        else:
            # build local sim matrix
            sub_sim = sim[np.ix_(inds, inds)]
            labels_local = union_find(sub_sim, threshold=0.6)
            # reindex so each top cluster's subclusters start at 0
            for local_idx, sub_lab in zip(inds, labels_local):
                sub_labels[local_idx] = (lab, sub_lab)
            # name each subcluster
            grouped = {}
            for idx, sub_lab in zip(inds, labels_local):
                grouped.setdefault(sub_lab, []).append(idx)
            for sub_lab, sub_inds in grouped.items():
                sub_names[(lab, sub_lab)] = name_cluster(keywords, sub_inds, stop_words)
    progress.progress(step/steps)
    step += 1

    # 6) Assemble DataFrame
    status_text.text(f"{step}/{steps} – Assembling output…")
    rows = []
    for i, kw in enumerate(keywords):
        top   = top_names[top_labels[i]]
        sub   = sub_names[sub_labels[i]]
        rows.append({"Keyword": kw, "Cluster": top, "Subcluster": sub})
    df = pd.DataFrame(rows)
    progress.progress(step/steps)
    step += 1

    # 7) Done
    status_text.text("Done!")
    progress.progress(1.0)
    return df

def main():
    st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Hierarchical Auto-Clustering Tool")
    st.markdown("""
    • Level 1: BERT + threshold clustering (top clusters)  
    • Level 2: BERT + threshold within each top cluster (subclusters)  
    • Named by core token + shortest phrase, entirely automatic.
    """)

    raw = st.text_area("🔤 Keywords (one per line):", height=300)
    keywords = [k for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        model = load_model()
        progress = st.progress(0.0)
        status_text = st.empty()
        try:
            df = cluster_and_hierarchy(keywords, model, progress, status_text)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="clusters.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

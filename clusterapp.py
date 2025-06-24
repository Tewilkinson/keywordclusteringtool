import streamlit as st
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

# — Cache the model with the new API —
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def cluster_and_name(keywords, model, progress, status_text):
    total_steps = 6
    step = 1

    # 1) Embed keywords
    status_text.text("1/6 – Embedding keywords…")
    emb = model.encode(keywords, normalize_embeddings=True)
    progress.progress(step / total_steps)
    step += 1

    # 2) Build cosine-similarity matrix
    status_text.text("2/6 – Building similarity matrix…")
    sim = np.dot(emb, emb.T)
    progress.progress(step / total_steps)
    step += 1

    # 3) Affinity Propagation clustering
    status_text.text("3/6 – Running Affinity Propagation…")
    ap = AffinityPropagation(affinity="precomputed", random_state=42)
    ap.fit(sim)
    labels = ap.labels_.copy()
    progress.progress(step / total_steps)
    step += 1

    # 4) Initial cluster groups
    status_text.text("4/6 – Grouping clusters…")
    clusters = {
        cid: [i for i, lab in enumerate(labels) if lab == cid]
        for cid in np.unique(labels)
    }
    progress.progress(step / total_steps)
    step += 1

    # 5) Merge any singleton clusters
    status_text.text("5/6 – Merging singleton noise…")
    cluster_means = {
        cid: emb[idxs].mean(axis=0)
        for cid, idxs in clusters.items()
    }
    for cid, idxs in list(clusters.items()):
        if len(idxs) == 1:
            lone_idx = idxs[0]
            best_other, best_sim = None, -1.0
            for ocid, mean_vec in cluster_means.items():
                if ocid == cid: continue
                sim_score = cosine_similarity(
                    emb[lone_idx : lone_idx + 1],
                    mean_vec.reshape(1, -1)
                )[0][0]
                if sim_score > best_sim:
                    best_sim, best_other = sim_score, ocid
            labels[lone_idx] = best_other

    # rebuild clusters
    clusters = {
        cid: [i for i, lab in enumerate(labels) if lab == cid]
        for cid in np.unique(labels)
    }
    progress.progress(step / total_steps)
    step += 1

    # 6) Name clusters by core token + shortest phrase
    status_text.text("6/6 – Naming clusters…")
    stop_words = {"best", "free", "online", "software", "generator"}
    cluster_names = {}
    for cid, idxs in clusters.items():
        tokens = []
        for i in idxs:
            for w in re.findall(r"\w+", keywords[i].lower()):
                if w not in stop_words:
                    tokens.append(w)
        if tokens:
            primary = max(set(tokens), key=tokens.count)
            candidates = [keywords[i] for i in idxs if primary in keywords[i].lower()]
            name = min(candidates, key=len) if candidates else keywords[idxs[0]]
        else:
            name = keywords[idxs[0]]
        cluster_names[cid] = name

    # Build final DataFrame
    df = pd.DataFrame({
        "Keyword": keywords,
        "Cluster": [cluster_names[lab] for lab in labels]
    })
    progress.progress(1.0)
    status_text.text("Done!")
    return df

def main():
    st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Auto-Clustering Keyword Tool")
    st.markdown("""
    Paste your keywords (one per line).  
    BERT + Affinity Propagation will discover clusters,  
    merge singleton noise, and name each cluster by its core phrase.
    """)

    raw = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        model = load_model()

        # set up progress bar & status text
        progress = st.progress(0.0)
        status_text = st.empty()

        try:
            df = cluster_and_name(keywords, model, progress, status_text)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="clusters.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Clustering error: {e}")

if __name__ == "__main__":
    main()

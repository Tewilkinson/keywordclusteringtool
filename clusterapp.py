import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# — Cache the model —
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def hierarchical_cluster_and_name(keywords, model, progress, status_text, level1_threshold=0.6):
    """
    Perform 2-level clustering:
      1) Agglomerative clustering for broad 'Level 1' groups based on distance threshold.
      2) AffinityPropagation within each Level 1 group to get finer 'Subclusters'.
    Returns a DataFrame with columns: Keyword, Level 1, Subcluster.
    """
    total_steps = 8
    step = 1

    # 1) Embed keywords
    status_text.text(f"{step}/{total_steps} – Embedding keywords…")
    emb = model.encode(keywords, normalize_embeddings=True)
    progress.progress(step / total_steps)
    step += 1

    # 2) Build similarity matrix
    status_text.text(f"{step}/{total_steps} – Building similarity matrix…")
    sim = np.dot(emb, emb.T)
    progress.progress(step / total_steps)
    step += 1

    # 3) Level 1 clustering (Agglomerative)
    status_text.text(f"{step}/{total_steps} – Level 1 clustering…")
    dist = 1 - sim  # cosine distance
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=level1_threshold,
        metric='precomputed',
        linkage='average'
    )
    lvl1_labels = agg.fit_predict(dist)
    progress.progress(step / total_steps)
    step += 1

    # 4) Name Level 1 clusters
    status_text.text(f"{step}/{total_steps} – Naming Level 1 clusters…")
    level1_clusters = {cid: [i for i, lab in enumerate(lvl1_labels) if lab == cid]
                       for cid in np.unique(lvl1_labels)}
    stop_words = {"best", "free", "online", "software", "generator"}
    level1_names = {}
    for cid, idxs in level1_clusters.items():
        tokens = []
        for i in idxs:
            tokens += [w for w in re.findall(r"\w+", keywords[i].lower()) if w not in stop_words]
        if tokens:
            primary = max(set(tokens), key=tokens.count)
            candidates = [keywords[i] for i in idxs if primary in keywords[i].lower()]
            level1_names[cid] = min(candidates, key=len) if candidates else keywords[idxs[0]]
        else:
            level1_names[cid] = keywords[idxs[0]]
    progress.progress(step / total_steps)
    step += 1

    # 5) Level 2 clustering (Affinity Propagation within each Level 1 group)
    status_text.text(f"{step}/{total_steps} – Level 2 clustering…")
    sub_label_pairs = [None] * len(keywords)

    for cid, idxs in level1_clusters.items():
        if len(idxs) > 1:
            sub_sim = sim[np.ix_(idxs, idxs)]
            ap = AffinityPropagation(affinity="precomputed", random_state=42)
            ap.fit(sub_sim)
            for j, idx in enumerate(idxs):
                sub_label_pairs[idx] = (cid, int(ap.labels_[j]))
        else:
            sub_label_pairs[idxs[0]] = (cid, 0)

    progress.progress(step / total_steps)
    step += 1

    # 6) Name subclusters
    status_text.text(f"{step}/{total_steps} – Naming subclusters…")
    clusters2 = {}
    for i, pair in enumerate(sub_label_pairs):
        clusters2.setdefault(pair, []).append(i)

    subcluster_names = {}
    for pair, idxs in clusters2.items():
        tokens = []
        for i in idxs:
            tokens += [w for w in re.findall(r"\w+", keywords[i].lower()) if w not in stop_words]
        if tokens:
            primary = max(set(tokens), key=tokens.count)
            candidates = [keywords[i] for i in idxs if primary in keywords[i].lower()]
            subcluster_names[pair] = min(candidates, key=len) if candidates else keywords[idxs[0]]
        else:
            subcluster_names[pair] = keywords[idxs[0]]
    progress.progress(step / total_steps)
    step += 1

    # 7) Build final DataFrame
    status_text.text(f"{step}/{total_steps} – Finalizing…")
    rows = []
    for i, kw in enumerate(keywords):
        lvl1 = level1_names[lvl1_labels[i]]
        sub = subcluster_names[sub_label_pairs[i]]
        rows.append({"Keyword": kw, "Level 1": lvl1, "Subcluster": sub})

    df = pd.DataFrame(rows)
    progress.progress(1.0)
    status_text.text("Done!")
    return df


def main():
    st.set_page_config(page_title="⚙️ Hierarchical Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Hierarchical Auto-Clustering Keyword Tool")
    st.markdown("""
    Paste your keywords (one per line).  
    BERT + 2-level clustering will discover broad intent groups (Level 1),  
    then finer subclusters within each (Level 2), naming each by core phrase.
    """
    )

    raw = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    threshold = st.slider("Level 1 distance threshold", 0.3, 1.0, 0.6, 0.05)

    if st.button("Cluster Keywords") and keywords:
        model = load_model()
        progress = st.progress(0.0)
        status_text = st.empty()

        try:
            df = hierarchical_cluster_and_name(keywords, model, progress, status_text, level1_threshold=threshold)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="hierarchical_clusters.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Clustering error: {e}")

if __name__ == "__main__":
    main()

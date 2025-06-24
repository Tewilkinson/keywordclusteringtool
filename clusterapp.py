import streamlit as st
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

# — replace experimental_singleton with st.cache for model-loading —
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def cluster_and_name(keywords, model):
    emb = model.encode(keywords, normalize_embeddings=True)
    sim = np.dot(emb, emb.T)

    ap = AffinityPropagation(affinity="precomputed", random_state=42)
    ap.fit(sim)
    labels = ap.labels_.copy()

    # initial clusters
    clusters = {
        cid: [i for i, lab in enumerate(labels) if lab == cid]
        for cid in np.unique(labels)
    }

    # merge singletons
    cluster_means = {cid: emb[idcs].mean(0) for cid, idcs in clusters.items()}
    for cid, idcs in list(clusters.items()):
        if len(idcs) == 1:
            i0 = idcs[0]
            best_other, best_sim = None, -1.0
            for oc, mean in cluster_means.items():
                if oc == cid: continue
                s = cosine_similarity(emb[i0 : i0+1], mean.reshape(1, -1))[0,0]
                if s > best_sim:
                    best_sim, best_other = s, oc
            labels[i0] = best_other

    # rebuild clusters
    clusters = {
        cid: [i for i, lab in enumerate(labels) if lab == cid]
        for cid in np.unique(labels)
    }

    # name clusters
    stop_words = {"best","free","online","software","generator"}
    cluster_names = {}
    for cid, idcs in clusters.items():
        tokens = []
        for i in idcs:
            for w in re.findall(r"\w+", keywords[i].lower()):
                if w not in stop_words:
                    tokens.append(w)
        if tokens:
            primary = max(set(tokens), key=tokens.count)
            cands = [keywords[i] for i in idcs if primary in keywords[i].lower()]
            name = min(cands, key=len) if cands else keywords[idcs[0]]
        else:
            name = keywords[idcs[0]]
        cluster_names[cid] = name

    return pd.DataFrame({
        "Keyword": keywords,
        "Cluster": [cluster_names[lab] for lab in labels]
    })

def main():
    st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Auto-Clustering Keyword Tool")
    st.markdown("""
    Paste your keywords (one per line).  
    BERT + Affinity Propagation will discover clusters,  
    merge singletons, and name each by its core phrase.
    """)

    raw = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        model = load_model()
        with st.spinner("Clustering…"):
            try:
                df = cluster_and_name(keywords, model)
                st.success("Done!")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", data=csv,
                                   file_name="clusters.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Clustering error: {e}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

# — Cache the model so it only loads once —
@st.experimental_singleton
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def cluster_and_name(keywords, model):
    # 1. Embed (normalized)
    emb = model.encode(keywords, normalize_embeddings=True)

    # 2. Cosine-sim matrix
    sim = np.dot(emb, emb.T)

    # 3. Affinity Propagation
    ap = AffinityPropagation(affinity="precomputed", random_state=42)
    ap.fit(sim)
    labels = ap.labels_.copy()

    # 4. Build initial clusters
    clusters = {
        cid: [i for i, lab in enumerate(labels) if lab == cid]
        for cid in np.unique(labels)
    }

    # 5. Merge singleton clusters
    cluster_means = {cid: emb[idcs].mean(0) for cid, idcs in clusters.items()}
    for cid, idcs in list(clusters.items()):
        if len(idcs) == 1:
            idx = idcs[0]
            best_other, best_sim = None, -1
            for ocid, mean_vec in cluster_means.items():
                if ocid == cid: continue
                score = cosine_similarity(emb[idx:idx+1], mean_vec.reshape(1, -1))[0,0]
                if score > best_sim:
                    best_sim, best_other = score, ocid
            labels[idx] = best_other

    # 6. Re-build clusters
    clusters = {
        cid: [i for i, lab in enumerate(labels) if lab == cid]
        for cid in np.unique(labels)
    }

    # 7. Name clusters by core token + shortest phrase
    stop_words = {"best","free","online","software","generator"}
    names = {}
    for cid, idcs in clusters.items():
        tokens = []
        for i in idcs:
            for w in re.findall(r"\w+", keywords[i].lower()):
                if w not in stop_words:
                    tokens.append(w)
        if tokens:
            primary = max(set(tokens), key=tokens.count)
            candidates = [keywords[i] for i in idcs if primary in keywords[i].lower()]
            name = min(candidates, key=len) if candidates else keywords[idcs[0]]
        else:
            name = keywords[idcs[0]]
        names[cid] = name

    # 8. Build result DataFrame
    df = pd.DataFrame({
        "Keyword": keywords,
        "Cluster": [names[lab] for lab in labels]
    })
    return df

def main():
    st.set_page_config(page_title="Keyword Auto-Cluster", layout="wide")
    st.title("🤖 Auto-Clustering Keyword Tool")
    st.markdown("""
    Paste your keywords (one per line).  
    BERT + Affinity Propagation will discover clusters, merge singletons,  
    and name each cluster by its core phrase.
    """)

    raw = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        model = load_model()
        try:
            with st.spinner("Clustering keywords… this may take a few seconds"):
                df = cluster_and_name(keywords, model)
            st.success("Done!")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="clusters.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error during clustering: {e}")

if __name__ == "__main__":
    main()

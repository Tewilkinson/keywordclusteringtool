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
    automatically discover & name clusters (singleton noise automatically merged).
    """)

    # 1) Input
    raw = st.text_area("🔤 Keywords:", height=300)
    keywords = [k.strip() for k in raw.splitlines() if k.strip()]

    if st.button("Cluster Keywords") and keywords:
        # 2) Embed + normalize
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(keywords, normalize_embeddings=True)

        # 3) Build cosine-similarity matrix
        sim = np.dot(emb, emb.T)

        # 4) Affinity Propagation
        ap = AffinityPropagation(affinity="precomputed", random_state=42)
        ap.fit(sim)
        labels = ap.labels_.copy()

        # 5) Initial clusters
        clusters = {
            cid: [i for i, lab in enumerate(labels) if lab == cid]
            for cid in np.unique(labels)
        }

        # 6) Merge singletons into nearest cluster
        #    Compute mean embedding per cluster
        cluster_means = {
            cid: emb[idxs].mean(axis=0)
            for cid, idxs in clusters.items()
        }
        for cid, idxs in list(clusters.items()):
            if len(idxs) == 1:
                lone_idx = idxs[0]
                # find best other cluster by similarity to its mean
                best_other = None
                best_sim = -1.0
                for other_cid, mean_vec in cluster_means.items():
                    if other_cid == cid:
                        continue
                    sim_score = cosine_similarity(
                        emb[lone_idx : lone_idx + 1],
                        mean_vec.reshape(1, -1)
                    )[0][0]
                    if sim_score > best_sim:
                        best_sim = sim_score
                        best_other = other_cid
                # reassign label
                labels[lone_idx] = best_other

        # 7) Rebuild clusters after merge
        clusters = {
            cid: [i for i, lab in enumerate(labels) if lab == cid]
            for cid in np.unique(labels)
        }

        # 8) Name each cluster by its core token + shortest phrase
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
                candidates = [
                    keywords[i] for i in idxs
                    if primary in keywords[i].lower()
                ]
                name = min(candidates, key=len) if candidates else keywords[idxs[0]]
            else:
                name = keywords[idxs[0]]
            cluster_names[cid] = name

        # 9) Build output DataFrame
        df = pd.DataFrame({
            "Keyword": keywords,
            "Cluster": [cluster_names[lab] for lab in labels]
        })

        # 10) Display & download
        st.success(f"→ {len(clusters)} clusters formed (singletons merged).")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name="clusters.csv", mime="text/csv")

if __name__ == "__main__":
    main()

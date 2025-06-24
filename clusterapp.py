import streamlit as st
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer

# Cache the model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def union_find(sim, threshold):
    N = sim.shape[0]
    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(N):
        for j in range(i+1, N):
            if sim[i,j] > threshold:
                union(i,j)
    return [find(i) for i in range(N)]

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

def cluster_two_levels(keywords, model, progress, status):
    steps = 5
    s = 1

    # 1) Embed
    status.text(f"{s}/{steps} – Embedding keywords…")
    emb = model.encode(keywords, normalize_embeddings=True)
    progress.progress(s/steps); s+=1

    # 2) Full‐matrix cosine similarity
    status.text(f"{s}/{steps} – Computing similarity…")
    sim = np.dot(emb, emb.T)
    progress.progress(s/steps); s+=1

    # 3) Level-1 clustering (broad)
    status.text(f"{s}/{steps} – Level-1 clustering…")
    lvl1_labels = union_find(sim, threshold=0.60)
    progress.progress(s/steps); s+=1

    # Name level-1 clusters
    lvl1_clusters = {}
    for i, lab in enumerate(lvl1_labels):
        lvl1_clusters.setdefault(lab, []).append(i)
    lvl1_names = {lab: name_cluster(keywords, inds) 
                  for lab, inds in lvl1_clusters.items()}

    # 4) Level-2 clustering (intent) within each lvl1
    status.text(f"{s}/{steps} – Level-2 clustering…")
    rows = []
    for lab, inds in lvl1_clusters.items():
        if len(inds)==1:
            # singleton → just itself
            sub_labels = [0]
            sub_names = {0: keywords[inds[0]]}
        else:
            sub_sim = sim[np.ix_(inds, inds)]
            sub_lbls = union_find(sub_sim, threshold=0.50)
            # group
            grouped = {}
            for idx, sl in zip(inds, sub_lbls):
                grouped.setdefault(sl, []).append(idx)
            sub_names = {sl: name_cluster(keywords, grp) 
                         for sl, grp in grouped.items()}
            sub_labels = sub_lbls

        # assemble rows
        for idx, sl in zip(inds, sub_labels):
            rows.append({
                "Keyword": keywords[idx],
                "Cluster": lvl1_names[lab],
                "Intent": sub_names[sl]
            })
    progress.progress(s/steps); s+=1

    # 5) Done
    status.text("Done!")
    pr

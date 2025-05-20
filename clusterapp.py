import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Embedding helper ---
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# --- Streamlit UI ---
st.set_page_config(page_title="Keyword Cluster App", layout="wide")
st.title("🔗 Auto Keyword Clustering Tool")

st.markdown("""
Paste your list of keywords below. We'll use GPT and embeddings to automatically group them into meaningful clusters.
""")

keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

if st.button("Cluster Keywords") and keyword_input:
    keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]

    st.info("Generating embeddings... This may take a moment.")
    embeddings = [get_embedding(k) for k in keywords]

    st.info("Clustering embeddings...")
    n_clusters = st.slider("Choose number of clusters", 2, min(15, len(keywords)), 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    df = pd.DataFrame({"Keyword": keywords, "Cluster ID": labels})

    st.info("Naming clusters using GPT...")
    cluster_names = {}
    for label in sorted(df["Cluster ID"].unique()):
        sample_keywords = df[df["Cluster ID"] == label]["Keyword"].head(5).tolist()
        prompt = (
            f"Name the common theme or product category of these keywords: {', '.join(sample_keywords)}.\n"
            f"Return only a short, specific cluster name. Be precise. Avoid generic terms like 'Snowflake Computing' or 'General Info'."
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        cluster_names[label] = response.choices[0].message.content.strip()

    df["Cluster Name"] = df["Cluster ID"].map(cluster_names)

    st.success("Done! View your clustered results below:")
    st.dataframe(df.sort_values("Cluster ID"))

    csv = df.to_csv(index=False)
    st.download_button("📥 Download Clustered CSV", data=csv, file_name="clustered_keywords.csv", mime="text/csv")

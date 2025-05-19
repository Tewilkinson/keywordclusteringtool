import streamlit as st
import pandas as pd
import openai
import numpy as np
from sklearn.cluster import KMeans
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Google Sheets Auth ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("gspread_credentials.json", scope)
gs_client = gspread.authorize(creds)

# --- Streamlit UI ---
st.set_page_config(page_title="Keyword Cluster App", layout="wide")
st.title("🔗 Keyword Clustering Tool")

keyword_input = st.text_area("Paste your keywords (one per line)", height=300)

if st.button("Cluster Keywords") and keyword_input:
    keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]

    st.info("Generating embeddings... This may take a moment.")
    embeddings = [get_embedding(k, engine="text-embedding-ada-002") for k in keywords]

    n_clusters = st.slider("Select number of clusters", 2, min(15, len(keywords)), 5)

    st.info("Clustering keywords...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    df = pd.DataFrame({"Keyword": keywords, "Cluster ID": labels})

    # Auto-name clusters using GPT
    st.info("Naming clusters using GPT...")
    cluster_names = {}
    for label in sorted(df["Cluster ID"].unique()):
        sample_keywords = df[df["Cluster ID"] == label]["Keyword"].head(5).tolist()
        prompt = f"Group the following keywords under a common product or topic name:\n{', '.join(sample_keywords)}\n\nReturn only the cluster name."
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        cluster_names[label] = response["choices"][0]["message"]["content"].strip()

    df["Cluster Name"] = df["Cluster ID"].map(cluster_names)

    st.success("Done! View your clustered results below:")
    st.dataframe(df.sort_values("Cluster ID"))

    # Export to Google Sheets
    sheet_name = st.text_input("Enter Google Sheet name for export", value="Clustered Keywords")
    if st.button("Export to Google Sheets"):
        sheet = gs_client.create(sheet_name)
        sheet.share('', perm_type='anyone', role='writer')  # Public write for quick testing (adjust in prod)
        ws = sheet.sheet1
        ws.update([df.columns.values.tolist()] + df.values.tolist())
        st.success(f"Exported to Google Sheets: {sheet.url}")
        st.markdown(f"[Open Sheet]({sheet.url})")

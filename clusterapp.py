import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Streamlit UI ---
st.set_page_config(page_title="Keyword Classifier App", layout="wide")
st.title("🔍 Keyword Classification Tool")

st.markdown("""
Paste your list of keywords below, and define the list of categories you want to classify them into.
Each keyword will be assigned to the closest matching category using GPT.
""")

keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)
category_input = st.text_area("🏷️ Define your categories (comma-separated):", value="Snowflake, Snowflake Database, Snowflake Certification, Snowflake Pricing, Snowflake Training, Snowflake Cortex, Snowflake Software, Snowflake Financial Reports, Snowflake Summit")

if st.button("Classify Keywords") and keyword_input and category_input:
    keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]
    categories = [c.strip() for c in category_input.split(",") if c.strip()]

    df = pd.DataFrame({"Keyword": keywords})
    cluster_names = []

    with st.spinner("Classifying keywords with GPT..."):
        for kw in keywords:
            prompt = f"Assign the keyword '{kw}' to one of the following clusters: {', '.join(categories)}.\n\nJust return the best matching cluster name."
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            cluster = response.choices[0].message.content.strip()
            cluster_names.append(cluster)

    df["Assigned Cluster"] = cluster_names
    st.success("Classification complete!")
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download Classified CSV", data=csv, file_name="classified_keywords.csv", mime="text/csv")

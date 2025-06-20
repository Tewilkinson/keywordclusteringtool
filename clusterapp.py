import streamlit as st
import pandas as pd
from openai import OpenAI
import time

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Streamlit UI ---
st.set_page_config(page_title="Keyword Classifier", layout="wide")
st.title("🎯 Keyword Category Classifier")

st.markdown("""
Paste your keywords below. This tool will automatically assign each keyword to a specific product/topic category using GPT-4-turbo.
""")

keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

if st.button("Classify Keywords") and keyword_input:
    keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]
    df = pd.DataFrame({"Keyword": keywords})
    category_labels = []

    progress = st.progress(0)
    status_text = st.empty()

    try:
        for i, kw in enumerate(keywords):
            prompt = (
                f"You are a product taxonomy expert.\n"
                f"Classify the keyword: {kw}\n"
                f"Return only the clean category name that best fits the keyword.\n"
                f"Do NOT repeat the keyword in your answer.\n"
                f"Examples:\n"
                f"snowflake cost → Snowflake Pricing\n"
                f"snowflake certification → Snowflake Certification\n"
                f"snowflake course → Snowflake Education\n"
                f"snowflake earnings → Snowflake Financials\n"
                f"snowflake cortex → Snowflake Cortex\n"
                f"snowflake summit → Snowflake Summit\n"
                f"snowflake logo → Snowflake Branding\n"
                f"Return only the final category label, nothing else."
            )
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            category = response.choices[0].message.content.strip()
            category_labels.append(category)
            progress.progress((i + 1) / len(keywords))
            status_text.text(f"Classified {i + 1} of {len(keywords)}")
            time.sleep(0.5)

        df["Assigned Category"] = category_labels
        st.success("Classification complete!")
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("📥 Download Classified CSV", data=csv, file_name="classified_keywords.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred during classification: {str(e)}")

import streamlit as st
import pandas as pd
from openai import OpenAI

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Streamlit UI ---
st.set_page_config(page_title="Keyword Classifier", layout="wide")
st.title("🎯 Keyword Category Classifier")

st.markdown("""
Paste your keywords below. This tool will automatically assign each keyword to a specific product/topic category using GPT-4.
""")

keyword_input = st.text_area("🔤 Paste keywords (one per line):", height=300)

if st.button("Classify Keywords") and keyword_input:
    keywords = [k.strip() for k in keyword_input.splitlines() if k.strip()]

    df = pd.DataFrame({"Keyword": keywords})
    category_labels = []

    with st.spinner("Classifying each keyword with GPT..."):
        for kw in keywords:
            prompt = (
                f"Assign the keyword '{kw}' to the most appropriate product or topic category.\n"
                f"Use specific product groupings based on Snowflake’s offerings.\n"
                f"Examples:\n"
                f"- 'snowflake cost' → Snowflake Pricing\n"
                f"- 'snowflake certification' → Snowflake Certification\n"
                f"- 'snowflake course' → Snowflake Education\n"
                f"- 'snowflake earnings' → Snowflake Financials\n"
                f"- 'snowflake cortex' → Snowflake Cortex\n"
                f"- 'snowflake summit' → Snowflake Summit\n"
                f"- 'snowflake logo' → Snowflake Logo\n"
                f"Do NOT overgeneralize. Return only the most specific, correct category name."
            )
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            category = response.choices[0].message.content.strip()
            category_labels.append(category)

    df["Assigned Category"] = category_labels
    st.success("Classification complete!")
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download Classified CSV", data=csv, file_name="classified_keywords.csv", mime="text/csv")

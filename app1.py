import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------------- Page config ----------------
st.set_page_config(
    page_title="AI Feedback Analyzer",
    layout="wide"
)

st.title("AI-Powered Customer Feedback Analyzer")
st.caption(
    "Upload customer feedback, analyze sentiment, and get actionable recommendations using OpenAI."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Configuration")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your key is used only for this session and is not stored."
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

# ---------------- Validation ----------------
if not openai_key:
    st.info("Please enter your OpenAI API key to continue.")
    st.stop()

if uploaded_file is None:
    st.info("Please upload a dataset containing a Feedback column.")
    st.stop()

# ---------------- Load data ----------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if "Feedback" not in df.columns:
    st.error("Dataset must contain a column named 'Feedback'.")
    st.stop()

st.success(f"Dataset loaded successfully with {len(df)} records.")
st.dataframe(df.head(10))

# ---------------- OpenAI Client ----------------
client = OpenAI(api_key=openai_key)

# ---------------- Analysis ----------------
st.subheader("AI Feedback Analysis & Recommendations")

feedback_text = "\n".join(df["Feedback"].astype(str).tolist()[:100])

prompt = f"""
You are a senior business analyst.

Analyze the following customer feedback and provide:
1. Overall sentiment summary
2. Key recurring issues or themes
3. Positive highlights
4. Clear, actionable business recommendations

Customer Feedback:
{feedback_text}
"""

if st.button("Generate Insights"):
    with st.spinner("Analyzing feedback with AI..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert business analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )

            result = response.choices[0].message.content

            st.markdown("### 📊 Analysis Results")
            st.markdown(result)

        except Exception as e:
            st.error(f"OpenAI API Error: {e}")

# ---------------- Footer ----------------
st.divider()
st.caption("Built with Streamlit and OpenAI | Customer Feedback Intelligence")

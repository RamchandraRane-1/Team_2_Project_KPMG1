import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Decision Intelligence Analyzer",
    layout="wide"
)

st.title("Decision Intelligence Analyzer")
st.caption(
    "AI-Driven Behavioral Pattern Mining from Customer Feedback and Survey Responses"
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Configuration")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your API key is used only for this session."
    )

    decision_objective = st.selectbox(
        "Decision Objective",
        [
            "Improve Customer Retention",
            "Reduce Customer Complaints",
            "Enhance Product Experience",
            "Improve Service Quality"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV or Excel)",
        type=["csv", "xlsx"]
    )

# ---------------- Validation ----------------
if not openai_key:
    st.info("Please enter your OpenAI API key to proceed.")
    st.stop()

if uploaded_file is None:
    st.info("Please upload a dataset containing a 'Feedback' column.")
    st.stop()

# ---------------- Load Dataset ----------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read dataset: {e}")
    st.stop()

if "Feedback" not in df.columns:
    st.error("Dataset must contain a column named 'Feedback'.")
    st.stop()

st.success(f"Dataset loaded successfully ({len(df)} records)")
st.dataframe(df.head(10))

# ---------------- OpenAI Client ----------------
client = OpenAI(api_key=openai_key)

# ---------------- Analysis Section ----------------
st.divider()
st.subheader("Behavioral Pattern & Decision Intelligence Analysis")

feedback_text = "\n".join(df["Feedback"].astype(str).tolist()[:120])

prompt = f"""
You are a Decision Intelligence expert and senior business analyst.

Decision Objective:
{decision_objective}

Analyze the following customer feedback and provide structured output with:

1. Behavioral Patterns Identified
   - Repeated behaviors, emotions, or user reactions
   - Group them into categories (e.g., Friction, Satisfaction, Churn Risk)

2. Pattern Frequency & Risk Level
   - Indicate whether each pattern is High / Medium / Low frequency
   - Assign a business risk level (High / Medium / Low)

3. Decision Mapping
   - For each major pattern, suggest:
     • Recommended decision/action
     • Business area impacted
     • Priority level

4. Strategic Recommendations
   - Clear, actionable steps aligned with the decision objective

Customer Feedback:
{feedback_text}
"""

if st.button("Run Decision Intelligence Analysis"):
    with st.spinner("Analyzing behavioral patterns and decision signals..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in decision intelligence and behavioral analytics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            result = response.choices[0].message.content

            st.markdown("### 🧠 Decision Intelligence Output")
            st.markdown(result)

        except Exception as e:
            st.error(f"OpenAI API Error: {e}")

# ---------------- Footer ----------------
st.divider()
st.caption(
    "Decision Intelligence Analyzer | AI-Driven Behavioral Pattern Mining | Built with Streamlit & OpenAI"
)

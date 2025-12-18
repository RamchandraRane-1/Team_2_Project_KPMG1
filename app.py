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
    "AI-Driven Behavioral Pattern Mining from Feedback and Survey Responses"
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
    st.info("Please upload a dataset containing 'Feedback' and 'Genre' columns.")
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

required_cols = ["Feedback", "Genre"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"Dataset must contain columns: {missing_cols}")
    st.stop()

st.success(f"Dataset loaded successfully ({len(df)} records)")
st.dataframe(df.head(10))

# ---------------- Genre-wise Visualization ----------------
st.divider()
st.subheader("Genre-wise Feedback Distribution")

genre_counts = df["Genre"].value_counts().reset_index()
genre_counts.columns = ["Genre", "Feedback Count"]

st.bar_chart(
    genre_counts.set_index("Genre")
)

# ---------------- OpenAI Client ----------------
client = OpenAI(api_key=openai_key)

# ---------------- Genre-wise Feedback Selection ----------------
st.divider()
st.subheader("Genre-wise Behavioral Analysis")

selected_genre = st.selectbox(
    "Select Genre for Deep Analysis",
    sorted(df["Genre"].unique())
)

genre_df = df[df["Genre"] == selected_genre]

st.write(f"Feedback records for **{selected_genre}**: {len(genre_df)}")
st.dataframe(genre_df[["Genre", "Feedback"]].head(10))

# ---------------- Decision Intelligence Prompt ----------------
feedback_text = "\n".join(genre_df["Feedback"].astype(str).tolist()[:100])

prompt = f"""
You are a Decision Intelligence expert and behavioral analytics specialist.

Decision Objective:
{decision_objective}

Genre:
{selected_genre}

Analyze the customer feedback below and provide:

1. Behavioral Patterns Identified
   - Repeated behaviors, emotions, or user reactions
   - Group them into categories (Friction, Satisfaction, Churn Risk)

2. Pattern Frequency & Business Risk
   - High / Medium / Low frequency
   - Business risk level

3. Decision Mapping
   - Recommended action
   - Business area impacted
   - Priority level

4. Strategic Recommendations
   - Clear, actionable steps aligned to the decision objective

Customer Feedback:
{feedback_text}
"""

# ---------------- Run Analysis ----------------
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
    "Decision Intelligence Analyzer | Behavioral Pattern Mining | Visual Analytics | Streamlit & OpenAI"
)

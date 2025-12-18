import streamlit as st
import pandas as pd
import altair as alt
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
        type="password"
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
    st.info("Enter OpenAI API key to continue.")
    st.stop()

if uploaded_file is None:
    st.info("Upload a dataset containing 'Feedback' and 'Genre' columns.")
    st.stop()

# ---------------- Load Dataset ----------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"File read error: {e}")
    st.stop()

required_cols = ["Feedback", "Genre"]
if any(col not in df.columns for col in required_cols):
    st.error("Dataset must contain 'Feedback' and 'Genre' columns.")
    st.stop()

st.success(f"Dataset loaded successfully ({len(df)} records)")

# ---------------- KPI Section ----------------
st.divider()
st.subheader("Executive Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Feedback Records", len(df))
col2.metric("Total Genres", df["Genre"].nunique())
col3.metric("Avg Feedback Length", int(df["Feedback"].astype(str).str.len().mean()))

# ---------------- Visualization 1: Genre Distribution ----------------
st.divider()
st.subheader("Genre-wise Feedback Volume")

genre_counts = df["Genre"].value_counts().reset_index()
genre_counts.columns = ["Genre", "Feedback Count"]

chart1 = alt.Chart(genre_counts).mark_bar().encode(
    x=alt.X("Genre:N", sort="-y"),
    y="Feedback Count:Q",
    tooltip=["Genre", "Feedback Count"]
)

st.altair_chart(chart1, use_container_width=True)

# ---------------- Genre Selection ----------------
st.divider()
st.subheader("Detailed Genre-wise Feedback Analysis")

selected_genre = st.selectbox(
    "Select Genre",
    sorted(df["Genre"].unique())
)

genre_df = df[df["Genre"] == selected_genre]

st.markdown(f"### 📌 {selected_genre} Overview")
st.write(f"Total feedback records: **{len(genre_df)}**")

# ---------------- Visualization 3: Feedback Length Distribution ----------------
chart3 = alt.Chart(genre_df).mark_bar().encode(
    x=alt.X("Feedback_Length:Q", bin=alt.Bin(maxbins=30), title="Feedback Length"),
    y=alt.Y("count()", title="Number of Feedbacks"),
    tooltip=["count()"]
)

st.altair_chart(chart3, use_container_width=True)

# ---------------- Sample Feedback ----------------
st.subheader("Sample Feedback (Preview)")
st.dataframe(genre_df[["Feedback"]].head(10))

# ---------------- OpenAI Client ----------------
client = OpenAI(api_key=openai_key)

# ---------------- Decision Intelligence Prompt ----------------
feedback_text = "\n".join(genre_df["Feedback"].astype(str).tolist()[:120])

prompt = f"""
You are a Decision Intelligence expert.

Decision Objective:
{decision_objective}

Genre:
{selected_genre}

Analyze the customer feedback below and provide:

1. Behavioral Patterns Identified
2. Pattern Frequency and Business Risk
3. Decision Mapping (action, impacted area, priority)
4. Strategic Recommendations aligned with the decision objective

Customer Feedback:
{feedback_text}
"""

# ---------------- Run AI Analysis ----------------
if st.button("Generate Decision Intelligence Insights"):
    with st.spinner("Analyzing behavioral patterns..."):
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

            st.divider()
            st.markdown("## 🧠 Decision Intelligence Output")
            st.markdown(result)

        except Exception as e:
            st.error(f"OpenAI Error: {e}")

# ---------------- Footer ----------------
st.divider()
st.caption(
    "Decision Intelligence Analyzer | Multi-Visualization Dashboard | Streamlit & OpenAI"
)


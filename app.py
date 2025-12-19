import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Decision Intelligence Dashboard", layout="wide")

st.title("Decision Intelligence Dashboard")
st.caption("AI-Driven Behavioral Pattern Mining from Customer Feedback")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Configuration")

    openai_key = st.text_input("OpenAI API Key", type="password")

    decision_objective = st.selectbox(
        "Decision Objective",
        [
            "Improve Customer Retention",
            "Reduce Customer Complaints",
            "Enhance Content Experience",
            "Improve Service Quality"
        ]
    )

    uploaded_file = st.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx"])

# ---------------- Validation ----------------
if not openai_key:
    st.info("Enter OpenAI API key to continue.")
    st.stop()

if uploaded_file is None:
    st.info("Upload the Netflix feedback dataset to continue.")
    st.stop()

# ---------------- Load Data ----------------
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

required_cols = ["Feedback", "Genre", "Content Type", "Year"]
if any(col not in df.columns for col in required_cols):
    st.error("Dataset structure does not match expected format.")
    st.stop()

st.success(f"Dataset loaded with {len(df)} records")

# ---------------- Simple Sentiment Classification ----------------
def classify_sentiment(text):
    text = str(text).lower()
    if any(w in text for w in ["great", "excellent", "amazing", "engaged", "loved"]):
        return "Positive"
    if any(w in text for w in ["bad", "poor", "rushed", "slow", "worst", "issue"]):
        return "Negative"
    return "Neutral"

df["Sentiment"] = df["Feedback"].apply(classify_sentiment)

# ================= DASHBOARD =================
st.divider()
st.subheader("📊 Executive Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Feedback Records", len(df))
col2.metric("Total Genres", df["Genre"].nunique())
col3.metric("Content Types", df["Content Type"].nunique())

# ---------------- Visualization 1: Genre-wise Sentiment ----------------
st.subheader("Genre-wise Sentiment Distribution")

genre_sentiment = (
    df.groupby(["Genre", "Sentiment"])
    .size()
    .reset_index(name="Count")
)

chart1 = alt.Chart(genre_sentiment).mark_bar().encode(
    x=alt.X("Genre:N", sort="-y"),
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Genre", "Sentiment", "Count"]
)

st.altair_chart(chart1, use_container_width=True)

# ---------------- Visualization 2: Content Type vs Sentiment ----------------
st.subheader("Content Type vs Sentiment")

content_sentiment = (
    df.groupby(["Content Type", "Sentiment"])
    .size()
    .reset_index(name="Count")
)

chart2 = alt.Chart(content_sentiment).mark_bar().encode(
    x="Content Type:N",
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Content Type", "Sentiment", "Count"]
)

st.altair_chart(chart2, use_container_width=True)

# ---------------- Visualization 3: Year-wise Sentiment Trend ----------------
st.subheader("Year-wise Sentiment Trend")

year_sentiment = (
    df.groupby(["Year", "Sentiment"])
    .size()
    .reset_index(name="Count")
)

chart3 = alt.Chart(year_sentiment).mark_line(point=True).encode(
    x="Year:O",
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Year", "Sentiment", "Count"]
)

st.altair_chart(chart3, use_container_width=True)

# ================= GENRE-WISE DEEP DIVE =================
st.divider()
st.subheader("🔍 Genre-wise Detailed Feedback Analysis")

selected_genre = st.selectbox("Select Genre", sorted(df["Genre"].unique()))
genre_df = df[df["Genre"] == selected_genre]

st.write(f"Total feedback records: {len(genre_df)}")

# ---------------- Genre Sentiment Breakdown ----------------
genre_sent = genre_df["Sentiment"].value_counts().reset_index()
genre_sent.columns = ["Sentiment", "Count"]

chart4 = alt.Chart(genre_sent).mark_bar().encode(
    x="Sentiment:N",
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Sentiment", "Count"]
)

st.altair_chart(chart4, use_container_width=True)

# ---------------- Sample Feedback ----------------
st.subheader("Representative Feedback Samples")
st.dataframe(genre_df[["Name", "Content Type", "Feedback", "Sentiment"]].head(10))

# ================= DECISION INTELLIGENCE =================
client = OpenAI(api_key=openai_key)

feedback_text = "\n".join(genre_df["Feedback"].astype(str).tolist()[:120])

prompt = f"""
You are a Decision Intelligence expert.

Decision Objective:
{decision_objective}

Genre:
{selected_genre}

Analyze the feedback and provide:
1. Key behavioral patterns
2. Business risks and opportunities
3. Decision mapping (action + priority)
4. Strategic recommendations

Feedback:
{feedback_text}
"""

if st.button("Generate Decision Intelligence Insights"):
    with st.spinner("Generating AI-driven insights..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in decision intelligence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        st.divider()
        st.markdown("## 🧠 Decision Intelligence Output")
        st.markdown(response.choices[0].message.content)

# ---------------- Footer ----------------
st.divider()
st.caption("Decision Intelligence Dashboard | Streamlit & OpenAI")

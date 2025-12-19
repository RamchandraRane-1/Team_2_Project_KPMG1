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

# ---------------- OpenAI Client ----------------
client = OpenAI(api_key=openai_key)

# ---------------- Sentiment Classification (Lightweight, AI-assisted) ----------------
def classify_sentiment(text):
    text = str(text).lower()
    if any(word in text for word in ["love", "great", "excellent", "amazing", "satisfied"]):
        return "Positive"
    if any(word in text for word in ["bad", "poor", "terrible", "worst", "delay", "issue"]):
        return "Negative"
    return "Neutral"

df["Sentiment"] = df["Feedback"].apply(classify_sentiment)

# ---------------- DASHBOARD: OVERALL SENTIMENT ----------------
st.divider()
st.subheader("Overall Sentiment Overview")

sentiment_counts = df["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

chart_overall = alt.Chart(sentiment_counts).mark_bar().encode(
    x="Sentiment:N",
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Sentiment", "Count"]
)

st.altair_chart(chart_overall, use_container_width=True)

# ---------------- DASHBOARD: GENRE-WISE SENTIMENT ----------------
st.divider()
st.subheader("Genre-wise Sentiment Distribution")

genre_sentiment = (
    df.groupby(["Genre", "Sentiment"])
    .size()
    .reset_index(name="Count")
)

chart_genre = alt.Chart(genre_sentiment).mark_bar().encode(
    x=alt.X("Genre:N", sort="-y"),
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Genre", "Sentiment", "Count"]
).properties(height=400)

st.altair_chart(chart_genre, use_container_width=True)

# ---------------- GENRE SELECTION ----------------
st.divider()
st.subheader("Detailed Genre-wise Decision Intelligence")

selected_genre = st.selectbox(
    "Select Genre for Deep Analysis",
    sorted(df["Genre"].unique())
)

genre_df = df[df["Genre"] == selected_genre]

# ---------------- GENRE-SPECIFIC SENTIMENT ----------------
st.markdown(f"### Sentiment Distribution: {selected_genre}")

genre_sentiment_dist = (
    genre_df["Sentiment"]
    .value_counts()
    .reset_index()
)
genre_sentiment_dist.columns = ["Sentiment", "Count"]

chart_genre_sent = alt.Chart(genre_sentiment_dist).mark_bar().encode(
    x="Sentiment:N",
    y="Count:Q",
    color="Sentiment:N",
    tooltip=["Sentiment", "Count"]
)

st.altair_chart(chart_genre_sent, use_container_width=True)

# ---------------- SAMPLE FEEDBACK ----------------
st.subheader("Representative Feedback Samples")
st.dataframe(genre_df[["Feedback", "Sentiment"]].head(10))

# ---------------- DECISION INTELLIGENCE PROMPT ----------------
feedback_text = "\n".join(genre_df["Feedback"].astype(str).tolist()[:120])

prompt = f"""
You are a Decision Intelligence expert.

Decision Objective:
{decision_objective}

Genre:
{selected_genre}

Analyze the customer feedback below and provide:

1. Key Behavioral Patterns (recurring behaviors and signals)
2. Risk Assessment (business impact of these patterns)
3. Decision Mapping (recommended actions, priority)
4. Strategic Recommendations aligned to the objective

Customer Feedback:
{feedback_text}
"""

# ---------------- RUN AI ANALYSIS ----------------
if st.button("Generate Decision Intelligence Insights"):
    with st.spinner("Generating decision intelligence insights..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in decision intelligence and behavioral analytics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            st.divider()
            st.markdown("## 🧠 Decision Intelligence Output")
            st.markdown(response.choices[0].message.content)

        except Exception as e:
            st.error(f"OpenAI Error: {e}")

# ---------------- Footer ----------------
st.divider()
st.caption(
    "Decision Intelligence Analyzer | AI-Driven Dashboard | Streamlit & OpenAI"
)

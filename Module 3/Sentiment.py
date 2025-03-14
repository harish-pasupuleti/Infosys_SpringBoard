import streamlit as st
import pandas as pd
import os
from pinecone import Pinecone
from langchain_together import TogetherEmbeddings
from together import Together

# 🎯 Set API keys (use environment variables for security)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hotel-reviews"

if not TOGETHER_API_KEY or not PINECONE_API_KEY:
    st.error("🚨 Error: API keys are missing! Set them as environment variables.")
    st.stop()

# 🔑 Set environment variable for Together API
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# 📂 Load dataset
try:
    df = pd.read_excel('reviews_data.xlsx')
    required_columns = {"review_id", "Review", "Rating", "review_date"}
    if not required_columns.issubset(df.columns):
        st.error(f"🚨 Missing required columns: {required_columns - set(df.columns)}")
        st.stop()
except FileNotFoundError:
    st.error("📂 Error: 'reviews_data.xlsx' not found!")
    st.stop()

# 🌎 Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        st.error(f"⚠️ Error: Pinecone index '{PINECONE_INDEX_NAME}' does not exist! Create it first.")
        st.stop()
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"❌ Pinecone initialization failed: {e}")
    st.stop()

# 🤖 Initialize Together AI Embeddings
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
client = Together()

# 🎨 Streamlit UI
st.title("🏨 Hotel Customer Sentiment Analysis 📊")
st.markdown("Analyze customer feedback with AI-powered sentiment analysis! 💬✨")

query = st.text_input("🔍 Enter a query about customer reviews:", "How is the food quality?")
start_date = st.date_input("📅 Start Date")
end_date = st.date_input("📅 End Date")
rating_filter = st.slider("⭐ Select Rating Filter", 1, 10, (1, 10))

if st.button("🚀 Analyze Sentiment"):
    try:
        query_embedding = embeddings.embed_query(query)

        # 🗓️ Convert dates to YYYYMMDD format for filtering
        start_date_str = int(start_date.strftime('%Y%m%d'))
        end_date_str = int(end_date.strftime('%Y%m%d'))

        # 🔍 Query Pinecone Index
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            filter={
                "Rating": {"$gte": rating_filter[0], "$lte": rating_filter[1]},
                "review_date": {"$gte": start_date_str, "$lte": end_date_str}
            }
        )

        # 📌 Extract Matching Results
        matches = results.get("matches", [])
        if not matches:
            st.warning("⚠️ No reviews found matching the criteria.")
        else:
            matched_ids = [int(match["metadata"].get("review_id", -1)) for match in matches if "metadata" in match]
            matched_ids = [mid for mid in matched_ids if mid != -1]  # Remove invalid IDs
            req_df = df[df["review_id"].isin(matched_ids)]
            concatenated_reviews = " ".join(req_df["Review"].tolist())

            # 📜 Generate Sentiment Summary
            response = client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[
                    {"role": "user", "content": f"""
                        Provide a concise summary of customer sentiment based on the following reviews:
                        {concatenated_reviews}.
                        Focus specifically on the manager's query: '{query}'.
                        Keep it short and objective. Do not mention the hotel's name.
                    """}
                ]
            )

            # 📢 Display Results
            st.subheader("📌 Sentiment Summary")
            st.success(response.choices[0].message.content)

            st.subheader("📝 Matched Reviews")
            st.dataframe(req_df[["Review"]])
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

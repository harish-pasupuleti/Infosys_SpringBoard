import pandas as pd
import streamlit as st
import plotly.express as px
from pymongo import MongoClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ✅ Step 1: Connect to MongoDB and Fetch Data (with error handling)
try:
    client = MongoClient("mongodb+srv://harishpasupulet")
    db = client["HotelAnalytics"]
    
    booking_df = pd.DataFrame(list(db["BookingData"].find()))
    dining_df = pd.DataFrame(list(db["DiningInfo"].find()))
    reviews_df = pd.DataFrame(list(db["ReviewsData"].find()))
    
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# ✅ Step 2: Data Preprocessing (handle missing values)
for df in [booking_df, dining_df, reviews_df]:
    df.fillna("Unknown", inplace=True)

# Convert Dates
booking_df["check_in_date"] = pd.to_datetime(booking_df["check_in_date"], errors="coerce")
booking_df["check_out_date"] = pd.to_datetime(booking_df["check_out_date"], errors="coerce")
booking_df.dropna(subset=["check_in_date", "check_out_date"], inplace=True)
booking_df["length_of_stay"] = (booking_df["check_out_date"] - booking_df["check_in_date"]).dt.days

dining_df["check_in_date"] = pd.to_datetime(dining_df["check_in_date"], errors="coerce")
reviews_df["Rating"] = pd.to_numeric(reviews_df["Rating"], errors="coerce")

# ✅ Step 3: Streamlit App Layout
st.set_page_config(page_title="Hotel Analytics Dashboard", layout="wide")
st.title("🏨 Hotel Analytics Dashboard")

# Sidebar for Navigation
menu = st.sidebar.radio("📊 Select Dashboard", ["Hotel Bookings", "Dining Insights", "Reviews Analysis"])

# ✅ Step 4: Hotel Bookings Dashboard
if menu == "Hotel Bookings":
    st.header("📅 Hotel Booking Insights")

    # 🔹 Line Chart: Bookings Trend Over Time
    booking_trend_df = booking_df.groupby("check_in_date").size().reset_index(name="count")
    fig1 = px.line(booking_trend_df, x="check_in_date", y="count", title="Bookings Trend Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    # 🔹 Bar Chart: Average Length of Stay (Weekly, Monthly)
    booking_df["week"] = booking_df["check_in_date"].dt.to_period("W").astype(str)
    booking_df["month"] = booking_df["check_in_date"].dt.to_period("M").astype(str)

    avg_stay_weekly = booking_df.groupby("week")["length_of_stay"].mean().reset_index()
    fig3 = px.bar(avg_stay_weekly, x="week", y="length_of_stay", title="Average Length of Stay (Weekly)")
    st.plotly_chart(fig3, use_container_width=True)

    avg_stay_monthly = booking_df.groupby("month")["length_of_stay"].mean().reset_index()
    fig4 = px.bar(avg_stay_monthly, x="month", y="length_of_stay", title="Average Length of Stay (Monthly)")
    st.plotly_chart(fig4, use_container_width=True)

# ✅ Step 5: Dining Insights Dashboard
elif menu == "Dining Insights":
    st.header("🍽️ Dining Insights")

    # 🔹 Pie Chart: Average Dining Cost by Cuisine (Fixed Column Name)
    if "Preferred Cusine" in dining_df.columns:
        dining_df.rename(columns={"Preferred Cusine": "Preferred Cuisine"}, inplace=True)
    
    if "Preferred Cuisine" in dining_df.columns:
        dining_cost_df = dining_df.groupby("Preferred Cuisine")["price_for_1"].mean().reset_index()
        fig1 = px.pie(dining_cost_df, names="Preferred Cuisine", values="price_for_1", title="Average Dining Cost by Cuisine")
        st.plotly_chart(fig1, use_container_width=True)

    # 🔹 Line Chart: Customer Count Over Time
    customer_count_df = dining_df.groupby(dining_df["check_in_date"].dt.to_period("M")).size().reset_index(name="count")
    customer_count_df["check_in_date"] = customer_count_df["check_in_date"].astype(str)
    fig2 = px.line(customer_count_df, x="check_in_date", y="count", title="Customer Count Over Time")
    st.plotly_chart(fig2, use_container_width=True)

# ✅ Step 6: Reviews Analysis Dashboard
elif menu == "Reviews Analysis":
    st.header("📝 Customer Reviews Analysis")

    # 🔹 Pie Chart: Sentiment Analysis
    reviews_df.dropna(subset=["Rating"], inplace=True)
    reviews_df["Sentiment"] = reviews_df["Rating"].apply(lambda x: "Positive" if x >= 7 else "Neutral" if x >= 4 else "Negative")
    sentiment_counts = reviews_df["Sentiment"].value_counts()
    fig1 = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="Sentiment Analysis")
    st.plotly_chart(fig1, use_container_width=True)

    # 🔹 Histogram: Rating Distribution
    fig2 = px.histogram(reviews_df, x="Rating", title="Rating Distribution", nbins=10)
    st.plotly_chart(fig2, use_container_width=True)

    # 🔹 Word Cloud: Customer Feedback
    if "ReviewText" in reviews_df.columns:
        st.subheader("📝 Customer Feedback Word Cloud")
        text = " ".join(reviews_df["ReviewText"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

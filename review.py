import streamlit as st
import pandas as pd
import os
from pinecone import Pinecone, ServerlessSpec
import together
import smtplib
from email.mime.text import MIMEText

# Initialize Pinecone
pc =  Pinecone(
    api_key='pcsk_3ZZU8b_MmaPjaJL3ZMmaUoqFNcJkFPy2M46ySn7QWvFpNgH8p2ZZKi7SjJxJ2W1UHTUeJ5'
)
index_name = "hotel-reviews"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )
index = pc.Index(index_name)

def send_email(review_details):
    sender_email = os.environ.get("EMAIL_SENDER")
    receiver_email = os.environ.get("EMAIL_MANAGER")
    password = os.environ.get("EMAIL_PASSWORD")
    
    subject = "New Customer Review"
    body = f"""
    Customer ID: {review_details['customer_id']}
    Room Number: {review_details['room_number']}
    Staying: {review_details['staying']}
    Review: {review_details['review']}
    """
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        st.success("Review emailed to manager!")
    except Exception as e:
        st.error(f"Error sending email: {e}")

# Load existing reviews dataset
def load_dataset():
    try:
        return pd.read_csv("reviews.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["customer_id", "review", "room_number", "staying"])

def save_review(data):
    df = load_dataset()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv("reviews.csv", index=False)

    # Embed the review and store in Pinecone
    try:
        embedding = together.embed_text(data["review"], api_key=os.environ.get("c0e3f0ff816fa5fdde9c8f22454a6f8777900ea945f7ec1928bb7903fc5b76e1"))
        index.upsert([(data["customer_id"], embedding, data)])
    except Exception as e:
        st.error(f"Error storing review in Pinecone: {e}")

# Streamlit UI
st.title("Hotel Review System")
customer_id = st.text_input("Customer ID")
review = st.text_area("Review")
room_number = st.text_input("Room Number")
staying = st.selectbox("Currently Staying?", ["Yes", "No"])

if st.button("Submit Review"):
    review_data = {
        "customer_id": customer_id,
        "review": review,
        "room_number": room_number,
        "staying": staying
    }
    save_review(review_data)
    send_email(review_data)
    st.success("Review submitted and emailed!")

import streamlit as st
from datetime import date
import pandas as pd
import random
import joblib
import xgboost
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb+srv://harishpasupuleti18:QzPqXVRXmYAYygHZ@cluster0.knp4m.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

st.title("🏨 Hotel Booking Form")

has_customer_id = st.radio("Do you have a Customer ID?", ("Yes", "No"))
customer_id = st.text_input("Enter your Customer ID", "") if has_customer_id == "Yes" else random.randint(10001, 99999)
st.write(f"Your generated Customer ID: {customer_id}")

# User Inputs
name = st.text_input("Enter your name", "")
checkin_date = st.date_input("Check-in Date", min_value=date.today())
checkout_date = st.date_input("Check-out Date", min_value=checkin_date)
age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
stayers = st.number_input("How many stayers in total?", min_value=1, max_value=3, step=1)
cuisine_options = ["South Indian", "North Indian", "Multi"]
preferred_cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
preferred_booking = st.selectbox("Do you want to book through points?", ["Yes", "No"])
special_requests = st.text_area("Any Special Requests? (Optional)", "")

if st.button("Submit Booking"):
    if name and customer_id:
        new_df = pd.DataFrame([{
            'customer_id': int(customer_id),
            'Preferred Cusine': preferred_cuisine,
            'age': age,
            'check_in_date': pd.to_datetime(checkin_date),
            'check_out_date': pd.to_datetime(checkout_date),
            'booked_through_points': 1 if preferred_booking == 'Yes' else 0,
            'number_of_stayers': stayers
        }])

        new_df['check_in_day'] = new_df['check_in_date'].dt.dayofweek
        new_df['check_out_day'] = new_df['check_out_date'].dt.dayofweek
        new_df['check_in_month'] = new_df['check_in_date'].dt.month
        new_df['check_out_month'] = new_df['check_out_date'].dt.month
        new_df['stay_duration'] = (new_df['check_out_date'] - new_df['check_in_date']).dt.days
        
        db = client["hotel_guests"]
        db["new_bookings"].insert_one(new_df.iloc[0].to_dict())

        customer_features = pd.read_excel('customer_features.xlsx')
        customer_dish = pd.read_excel('customer_dish.xlsx')
        cuisine_features = pd.read_excel('cuisine_features.xlsx')
        cuisine_dish = pd.read_excel('cuisine_dish.xlsx')

        new_df = new_df.merge(customer_features, on='customer_id', how='left')
        new_df = new_df.merge(cuisine_features, on='Preferred Cusine', how='left')
        new_df = new_df.merge(customer_dish, on='customer_id', how='left')
        new_df = new_df.merge(cuisine_dish, on='Preferred Cusine', how='left')

        new_df.drop(['customer_id', 'check_in_date', 'check_out_date'], axis=1, inplace=True)
        
        encoder = joblib.load('encoder.pkl')
        categorical_cols = ['Preferred Cusine', 'most_frequent_dish', 'cuisine_popular_dish']
        encoded_test_df = pd.DataFrame(encoder.transform(new_df[categorical_cols]), 
                                       columns=encoder.get_feature_names_out(categorical_cols))
        new_df = pd.concat([new_df.drop(columns=categorical_cols), encoded_test_df], axis=1)
        
        features = list(pd.read_excel('features.xlsx')[0])
        label_encoder = joblib.load('label_encoder.pkl')
        new_df = new_df[features]
        model = joblib.load('xgb_model_dining.pkl')
        
        y_pred_prob = model.predict_proba(new_df)
        dish_names = label_encoder.classes_
        top_3_dishes = dish_names[np.argsort(-y_pred_prob, axis=1)[:, :3]]
        
        st.success(f"✅ Booking Confirmed for {name} (Customer ID: {customer_id})!")
        st.write(f"**Check-in:** {checkin_date}")
        st.write(f"**Check-out:** {checkout_date}")
        st.write(f"**Preferred Cuisine:** {preferred_cuisine}")
        if special_requests:
            st.write(f"**Special Requests:** {special_requests}")
        
        dishes = [dish.lower() for dish in top_3_dishes[0]]
        thali_dishes = [dish for dish in dishes if "thali" in dish]
        other_dishes = [dish for dish in dishes if "thali" not in dish]
        
        st.write("Discounts for you!")
        if thali_dishes:
            st.write(f"Get 20% off on {', '.join(thali_dishes)}")
        if other_dishes:
            st.write(f"Get 15% off on {', '.join(other_dishes)}")
        st.write("Check your coupon code on your email!")
    else:
        st.warning("⚠️ Please enter your name and Customer ID to proceed!")

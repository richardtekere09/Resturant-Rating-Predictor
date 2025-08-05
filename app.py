import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(
    page_title="Restaurant Ratings Predictor",
    page_icon=":fork_and_knife:",
    layout="wide",
)

scaler = joblib.load("scaler_clean.pkl") 
st.title("Restaurant Ratings Predictor")
st.caption("This App help you to predict the rating of a restaurant based on its features.")

st.divider()
averagecost= st.number_input("PLease enter the estimated average Cost for Two", min_value=50, max_value=100000, value=1000, step=200) 
tablebooking= st.selectbox("Restaurant has Table Booking", ["Yes, No"])
onlinedilivery = st.selectbox("Restaurant has Online Delivery", ["Yes, No"])
pricerange = st.selectbox("Please select the Price Range (1 for Cheapest ,4  for Most Expensive)", ["1", "2", "3", "4"])
predictbutton = st.button("Predict Rating")

st.divider()
model= joblib_model = joblib.load('gridsrfr_model.pkl')
bookingstatus= 1 if tablebooking == "Yes" else 0
deliverystatus= 1 if onlinedilivery == "Yes" else 0

values= np.array([[averagecost, bookingstatus, deliverystatus, pricerange]])
my_x_values= np.array(values)
X=scaler.transform(my_x_values)

if predictbutton:
    st.snow()
    prediction= model.predict(X)
    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.5:
        st.write("Good")
    else:
        st.write("Excellent")
    st.write("Predicted Rating:", prediction)
   # st.success("The predicted rating is: {}".format(prediction))
else:
   st.info("Please enter the restaurant features and click on 'Predict Rating' to see the predicted rating.")

st.sidebar.header("About")
st.sidebar.info("This app uses a Random Forest Regressor model to predict restaurant ratings based on features such as average cost, table booking, online delivery, and price range.")
st.sidebar.text("Developed by Richard")  # Replace with your name or organization
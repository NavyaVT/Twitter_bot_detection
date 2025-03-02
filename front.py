import streamlit as st
from model import predict_bot
from twitter_X import getDetails

st.text("Welcome to bot detector")
st.title("Bot Detector")

userid = st.text_input("Enter twitter user name")

sub = st.button("SUBMIT")

if sub:
    if userid:
        try:
            tweet, retweet_count, mention_count, follower_count, verified = getDetails(userid)
            text = predict_bot(tweet, retweet_count, mention_count, follower_count, verified)
            if text == "Bot":
                st.error("This is a bot account")
            else:
                st.success("This is a human account")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a valid Twitter username.")

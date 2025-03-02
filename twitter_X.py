import tweepy
import os
from dotenv import load_dotenv
import time

load_dotenv("token.env")

SPARE_TOKEN = "AAAAAAAAAAAAAAAAAAAAAN8lzQEAAAAAeRLD%2FBwzn1wh%2F9k29BmBSmWRNAA%3D0ex7x0n3I84hJsFrZnZnpBorOeDpdkhbDjGWShxrORiR22g72g"

client = tweepy.Client(SPARE_TOKEN)

def getDetails(username):
    try:
        info = client.get_user(username=username, user_fields=["public_metrics","entities"])
        data = info.data
        tweet = client.get_users_tweets(id=data.id, tweet_fields=["public_metrics","entities"])

        print(f"Username: {data.username}")
        print(f"Name: {data.name}")
        print(f"ID: {data.id}")
        print(f"Bio: {data.description}")
        print(f"Location: {data.location}")
        print(f"Profile Image URL: {data.profile_image_url}")
        print(f"URL: {data.url}")
        print(f"Verified: {data.verified}")

        public = data.public_metrics
        if public:
            print(f"Followers Count: {public['followers_count']}")
            print(f"Following Count: {public['following_count']}")
            print(f"Tweet Count: {public['tweet_count']}")
            print(f"Listed Count: {public['listed_count']}")

        mentions_count = 0
        if tweet.data:
            for t in tweet.data:
                # print(f"Tweet ID: {t.id}")
                # print(f"Tweet Text: {t.text}")
                # print(f"Tweet Created At: {t.created_at}")

                # if t.public_metrics:
                    # print(f"Retweet Count: {t.public_metrics.get('retweet_count', 0)}")
                    # print(f"Reply Count: {t.public_metrics.get('reply_count', 0)}")
                    # print(f"Like Count: {t.public_metrics.get('like_count', 0)}")

                hashtags = t.entities.get('hashtags', []) if t.entities else []
                mentions = t.entities.get('mentions', []) if t.entities else []
                urls = t.entities.get('urls', []) if t.entities else []
                mentions_count += len(mentions)

                # print(f"Mentions Count: {mentions_count}")
                # print(f"Hashtags: {hashtags}")
                # print(f"Mentions: {mentions}")
                # print(f"URLs: {urls}")
                # print(f"Geo: {t.geo if t.geo else 'None'}")
                # print("-" * 30)
                if({data.verified}==None):
                    ver_count=0
                else:
                    ver_count={data.verified}
                print(t.text, {t.public_metrics.get('retweet_count', 0)}, mentions_count, {public['followers_count']}, ver_count, len(hashtags))
                return(t.text, {t.public_metrics.get('retweet_count', 0)}, mentions_count, {public['followers_count']}, ver_count, len(hashtags))
        
        else:
            print("No tweets found for this user.")
        
    except tweepy.TooManyRequests as e:
        reset_time = e.response.headers.get("x-rate-limit-reset")
        sleep = int(reset_time) - int(time.time()) + 5
        print(f"Error: {e}, sleeping for {sleep} seconds")
        time.sleep(sleep)

getDetails("timesofindia")
import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
consumer_key = os.getenv('TWITTER_API_KEY')
consumer_secret = os.getenv('TWITTER_API_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

# Replace with the username you want to fetch tweets from
username = 'Ai_Bot_Testing'

# Get user information
user = client.get_user(username=username)
user_id = user.data.id

# Fetch tweets from the user
tweets = client.get_users_tweets(
    id=user_id,
    exclude=['retweets'],
    tweet_fields=['id', 'text', 'created_at'],
    max_results=10  # You can adjust this number (max 100)
)

# Check if any tweets were returned
if tweets.data:
    for tweet in tweets.data:
        tweet_id = tweet.id
        tweet_text = tweet.text
        created_at = tweet.created_at

        print(f"Tweet ID: {tweet_id}")
        print(f"Tweet Text: {tweet_text}")
        print(f"Created At: {created_at}")
        print("-" * 50)
else:
    print("No tweets found.")

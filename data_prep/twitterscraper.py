import tweepy
import pandas as pd
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


usernames = ['intern', 'ChainLinkGod', 'SmokeyTheBera', 'VitalikButerin']  # Replace with actual usernames

tweets_data = []

for username in usernames:
    user = client.get_user(username=username)
    print('Getting user '+ str(user))
    user_id = user.data.id
    print('User ID: ' + str(user_id))

    paginator = tweepy.Paginator(
        client.get_users_tweets,
        id=user_id,
        exclude=['retweets'],  # include replies
        tweet_fields=['created_at', 'text', 'in_reply_to_user_id', 'conversation_id'],
        expansions=['referenced_tweets.id'],
        max_results=100
    )
    print('Storing Data')

    for tweet in paginator.flatten(limit=1000):
        is_reply = tweet.in_reply_to_user_id is not None

        tweets_data.append({
            'username': username,
            'user_id': user_id,
            'tweet_id': tweet.id,
            'created_at': tweet.created_at,
            'text': tweet.text,
            'is_reply': is_reply,
            'in_reply_to_user_id': tweet.in_reply_to_user_id,
            'conversation_id': tweet.conversation_id
        })

# Save to DataFrame
df = pd.DataFrame(tweets_data)
df.to_csv('tweets.csv', index=False, encoding='utf-8')


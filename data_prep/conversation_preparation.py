import pandas as pd

# Load the cleaned tweets
df = pd.read_csv('cleaned_tweets.csv')

# Dictionary to store tweets by tweet ID
tweet_dict = df.set_index('tweet_id').to_dict('index')

training_data = []

for index, row in df.iterrows():
    if row['is_reply']:
        # Get the previous tweet in the conversation
        in_reply_to_tweet_id = row['conversation_id']
        if in_reply_to_tweet_id in tweet_dict:
            previous_tweet = tweet_dict[in_reply_to_tweet_id]['clean_text']
            # Combine previous and current tweet
            conversation = f"[User]: {previous_tweet}\n[Bot]: {row['clean_text']}"
            training_data.append(conversation)
        else:
            # If previous tweet not found, use current tweet only
            training_data.append(f"[Bot]: {row['clean_text']}")
    else:
        # For original posts
        training_data.append(f"[Bot]: {row['clean_text']}")

# Save the training data
with open('training_data.txt', 'w', encoding='utf-8') as f:
    for item in training_data:
        f.write(f"{item}\n")

import pandas as pd
import re
import emoji

# Load the collected tweets
df = pd.read_csv('tweets.csv')

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\n', ' ', text)

    # Remove non-word characters except for spaces and emojis
    text = ''.join(c for c in text if c.isalnum() or c.isspace() or emoji.is_emoji(c))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Clean the tweet text
df['clean_text'] = df['text'].apply(clean_text)

# Remove duplicates and empty texts
df.drop_duplicates(subset=['clean_text'], inplace=True)
df = df[df['clean_text'].astype(bool)]

# Save the cleaned data
df.to_csv('cleaned_tweets.csv', index=False, encoding='utf-8')

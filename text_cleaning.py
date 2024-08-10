# %%
import re
import string

import duckdb
import pandas as pd

# %%
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# %%
connection = duckdb.connect('data/raw/NaijaSenti/train.duckdb')
connection.execute("SELECT tweet, label FROM data")
train_data = connection.fetchall()

connection = duckdb.connect('data/raw/NaijaSenti/validation.duckdb')
connection.execute("SELECT tweet, label FROM data")
validation_data = connection.fetchall()

connection = duckdb.connect('data/raw/NaijaSenti/test.duckdb')
connection.execute("SELECT tweet, label FROM data")
test_data = connection.fetchall()

data = train_data + validation_data + test_data
tweet_df = pd.DataFrame(data, columns=['text', 'sentiment'])
tweet_df['text'] = tweet_df['text'].astype(str).str.lower()
tweet_df['text'] = tweet_df['text'].apply(clean_text)
num_sentiment = tweet_df['sentiment'].to_list()
sentiment = []
for n in num_sentiment:
    if n == 0:
        sentiment.append('Positive')
    elif n == 1:
        sentiment.append('Neutral')
    else:
        sentiment.append('Negative')
tweet_df['sentiment'] = sentiment
tweet_df.to_csv('data/processed/NaijaSenti.csv', index=False)

# %%
train_df = pd.read_csv('data/raw/NollySenti/train.tsv', sep='\t')
dev_df = pd.read_csv('data/raw/NollySenti/dev.tsv', sep='\t')
test_df = pd.read_csv('data/raw/NollySenti/test.tsv', sep='\t')
review_df = pd.concat([train_df, dev_df, test_df])
review_df = review_df.rename(columns={'pcm_review': 'text'})
review_df['text'] = review_df['text'].astype(str).str.lower()
review_df['text'] = review_df['text'].apply(clean_text)
review_df['sentiment'] = review_df['sentiment'].str.title()
review_df.to_csv("data/processed/NollySenti.csv", index=False)

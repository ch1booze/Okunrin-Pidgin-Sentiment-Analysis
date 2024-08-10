# %%
import json
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# %%
# Dataset generation
naija_senti_df = pd.read_csv('data/processed/NaijaSenti.csv')
nolly_senti_df = pd.read_csv('data/processed/NollySenti.csv')
pidgin_sentiment_df = pd.concat([naija_senti_df, nolly_senti_df])
pidgin_sentiment_df = pidgin_sentiment_df.dropna()
print(len(pidgin_sentiment_df))
pidgin_sentiment_df.head()

# %%
with open('data/utils/replace.json', 'r') as file:
    replaceable = json.load(file)

def replace_text(text):
    text_list = text.split()
    new_text_list = []
    for t in text_list:
        if t != ' ':
            if replaceable.get(t) is not None:
                new_text_list.append(replaceable[t])
            else:
                new_text_list.append(t)

    return ' '.join(new_text_list)
    
pidgin_sentiment_df['text'] = pidgin_sentiment_df['text'].apply(replace_text)
pidgin_sentiment_df.to_csv('data/processed/PidginSenti.csv', index=False)

# %%
# Sentiment distribution
ax = pidgin_sentiment_df['sentiment'].value_counts() \
    .sort_index() \
    .plot(
        kind='bar',
        title='Positive v Neutral v Negative',
        figsize=(10, 5),
    )
ax.set_xlabel('Sentiment')
plt.show()

# %%
# Get list of words
text = " ".join(pidgin_sentiment_df["text"].to_list())
words = text.split(" ")

# %%
# Get most common words
counter = Counter(words)
most_common = counter.most_common(20)
most_common_dict = {i[0]: i[1] for i in most_common}
with open('data/analytics/most_common.json', 'w') as file:
    json.dump(most_common_dict, file)
most_common_dict

# %%
# Get vocabulary list
vocab = list(set(words))
with open('data/analytics/vocab.txt', 'w') as vocab_file:
    vocab_file.write('\n'.join(vocab))
len(vocab)

# %%
# Get length of sentences
text_lengths = pidgin_sentiment_df['text'].apply(lambda t: len(t))
longest_texts = text_lengths.nlargest(10)
shortest_texts = text_lengths.nsmallest(10)
most_common_length = text_lengths.value_counts().head(10)

# %%
# Get most common bigrams amd trigrams
bigrams = []
for text in pidgin_sentiment_df["text"]:
    text_list = text.split()
    for i in range(len(text_list) - 1):
        if text_list[i] != ' ' or text_list[i + 1] != ' ':
            bigrams.append(f'{text_list[i]} {text_list[i + 1]}')

bigram_counter = Counter(bigrams)
bigram_counter.most_common(10)

# %%
# Handling of imbalanced data
from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(random_state=0)
X = pidgin_sentiment_df['text'].to_numpy().reshape(-1, 1)
y = pidgin_sentiment_df['sentiment'].to_numpy()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

resampled_pidgin_sentiment_df = pd.DataFrame(columns=['text', 'sentiment'])
resampled_pidgin_sentiment_df['text'] = X_resampled.reshape(-1)
resampled_pidgin_sentiment_df['sentiment'] = y_resampled
resampled_pidgin_sentiment_df.to_csv('data/processed/ResampledPidginSenti.csv', index=False)
resampled_pidgin_sentiment_df.head(5)

ax = resampled_pidgin_sentiment_df['sentiment'].value_counts() \
    .sort_index() \
    .plot(
        kind='bar',
        title='Resampled Positive v Neutral v Negative',
        figsize=(10, 5),
    )
ax.set_xlabel('Sentiment')
plt.show()

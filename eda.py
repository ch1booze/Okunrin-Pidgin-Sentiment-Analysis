# %%
import duckdb
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

# %%
connection = duckdb.connect(database='datasets/pidgin_sentiment/train.duckdb', read_only=True)
connection.execute('SELECT tweet, label FROM data')
dataset = connection.fetchall()
dataset[0]

# %%
df = pd.DataFrame(dataset, columns=['pdg', 'label'])

# %%
# Sentiment Distribution
ax = df['label'].value_counts() \
    .sort_index() \
    .plot(
        kind='bar',
        title='Positive v Negative',
        figsize=(10, 5),
    )
ax.set_xlabel('Sentiment')
plt.show()

# %%
text = " ".join(df["pdg"])
words = text.split()
len(words)

from collections import Counter
import json
counter = Counter(words)
most_common = counter.most_common(20)
most_common_dict = {i[0]: i[1] for i in most_common}
with open("most_common.json", "w") as file:
    json.dump(most_common_dict, file)


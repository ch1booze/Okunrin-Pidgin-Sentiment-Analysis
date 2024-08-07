# %%
import duckdb
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gradio_client import Client
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# %%
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# %%
client = Client('https://nithub-ai-ehn-english-to-nigerian-pidgin-translator.hf.space')
pdg_to_eng = lambda pdg : client.predict('BBGM Model (PCM to EN)', pdg, api_name='/predict')
pdg_to_eng('Wetin Ifeanyi talk sef')
# %%
connection = duckdb.connect(database='datasets/pidgin_sentiment/train.duckdb', read_only=True)
connection.execute('SELECT tweet, label FROM data')
dataset = connection.fetchall()
dataset[0]

# %%
df = pd.DataFrame(dataset, columns=['pdg', 'label'])
df['label'] -= 1
neutrals = df[df['label'] == 0].reset_index(drop=True)
non_neutrals = df[df['label'] != 0] \
                .sample(n=200) \
                .reset_index(drop=True)
df = pd.concat([neutrals, non_neutrals]) \
        .reset_index(drop=True)

# %%
with open('result.txt', 'w') as file:
    file.write('')

for i in tqdm(range(len(df))):
    try:
        result = pdg_to_eng(df.loc[i, 'pdg'])
        with open('result.txt', 'a') as file:
            file.write(result[0] + '\n')
    except Exception as e:
        print(f'Error: {e}')

# %%
with open('result.txt', 'r') as file:
    eng = file.read().splitlines()
df['eng'] = eng

# %%
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
example = df["eng"][0]
tokens = nltk.word_tokenize(example)
print(tokens[:5])
tagged = nltk.pos_tag(tokens)
print(tagged[:5])
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# %%
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(example))
print(df['label'][0])

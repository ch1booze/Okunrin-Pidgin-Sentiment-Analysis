# %%
import numpy as np
import pandas as pd
from fast_ml.model_development import train_valid_test_split
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
from torch.nn.functional import softmax
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f'Device Availble: {DEVICE}')

# %%
import random

import duckdb

connection = duckdb.connect('datasets/pidgin_sentiment/test.duckdb')
connection.execute("SELECT tweet, label FROM data")
data = connection.fetchall()
random.shuffle(data)

pidgin = [x[0] for x in data]
num_labels = [x[1] for x in data]
labels = []
num_of_positive = 0
num_of_neutral = 0
num_of_negative = 0
for l in num_labels:
    if l == 0:
        l = "positive"
        num_of_positive += 1
    elif l == 1:
        l = "neutral"
        num_of_neutral += 1
    elif l == 2:
        l = "negative"
        num_of_negative += 1
    labels.append(l)

num_of_positive, num_of_neutral, num_of_negative

# %%
df = pd.DataFrame({"pidgin": pidgin, "label": labels})
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
print(le.classes_)
df.head()

# %%
class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        if bool(sentences):
            self.encodings = self.tokenizer(self.sentences,
                                            truncation = True,
                                            padding = True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        if self.labels == None:
            item['labels'] = None
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.sentences)


    def encode(self, x):
        return self.tokenizer(x, return_tensors = 'pt').to(DEVICE)

# %%
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_valid_test_split(df, target = 'label', train_size=0.8, valid_size=0.1, test_size=0.1)

# %%
train_texts = train_texts['pidgin'].to_list()
train_labels = train_labels.to_list()
val_texts = val_texts['pidgin'].to_list()
val_labels = val_labels.to_list()
test_texts = test_texts['pidgin'].to_list()
test_labels = test_labels.to_list()

train_dataset = DataLoader(train_texts, train_labels)
val_dataset = DataLoader(val_texts, val_labels)
test_dataset = DataLoader(test_texts, test_labels)

# %%
f1 = datasets.load_metric('f1')
accuracy = datasets.load_metric('accuracy')
precision = datasets.load_metric('precision')
recall = datasets.load_metric('recall')
def compute_metrics(eval_pred):
    metrics_dict = {}
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    metrics_dict.update(f1.compute(predictions = predictions, references = labels, average = 'macro'))
    metrics_dict.update(accuracy.compute(predictions = predictions, references = labels))
    metrics_dict.update(precision.compute(predictions = predictions, references = labels, average = 'macro'))
    metrics_dict.update(recall.compute(predictions = predictions, references = labels, average = 'macro'))
    return metrics_dict

# %%
id2label = {idx:label for idx, label in enumerate(le.classes_)}
label2id = {label:idx for idx, label in enumerate(le.classes_)}
config = AutoConfig.from_pretrained('distilbert-base-uncased',
                                    num_labels = 3,
                                    id2label = id2label,
                                    label2id = label2id)
model = AutoModelForSequenceClassification.from_config(config)
print(config)

# %%

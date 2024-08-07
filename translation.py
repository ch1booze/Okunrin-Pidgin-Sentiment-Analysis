# %%
import numpy as np
from transformer import Transformer

# %%
import duckdb

connection = duckdb.connect()
connection.execute("SELECT pidgin FROM 'datasets/eng_to_pdg/train.parquet'")
pidgin = connection.fetchall()
pidgin = [x[0] for x in pidgin]

connection = duckdb.connect()
connection.execute("SELECT english FROM 'datasets/eng_to_pdg/train.parquet'")
english = connection.fetchall()
english = [x[0] for x in english]

# %%
english_tokens = list(set("".join(english)))
pidgin_tokens = list(set("".join(pidgin)))

with open('english_tokens.txt', 'w') as file:
    file.writelines(english_tokens)

with open('pidgin_tokens.txt', 'w') as file:
    file.writelines(pidgin_tokens)

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

english_vocab = [START_TOKEN, PADDING_TOKEN, END_TOKEN, *english_tokens]
pidgin_vocab = [START_TOKEN, PADDING_TOKEN, END_TOKEN, *pidgin_tokens]
len(english_vocab), len(pidgin_vocab)

# %%
index_to_pidgin = {k:v for k,v in enumerate(pidgin_vocab)}
pidgin_to_index = {v:k for k,v in enumerate(pidgin_vocab)}
index_to_english = {k:v for k,v in enumerate(english_vocab)}
english_to_index = {v:k for k,v in enumerate(english_vocab)}

# %%
for i, j in zip(pidgin[:5], english[:5]):
    print(i, j)

# %%
PERCENTILE = 100
print(f"{PERCENTILE}th percentile length Pidgin: {np.percentile([len(x) for x in pidgin], PERCENTILE)}")
print(f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english], PERCENTILE)}")

# %%
max_sequence_length = 200

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)

valid_sentence_indicies = []
for index in range(len(pidgin)):
    pidgin_sentence, english_sentence = pidgin[index], english[index]
    if is_valid_length(pidgin_sentence, max_sequence_length) \
      and is_valid_length(english_sentence, max_sequence_length) \
      and is_valid_tokens(pidgin_sentence, pidgin_vocab):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(pidgin)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

# %%
pidgin_sentences = [pidgin[i] for i in valid_sentence_indicies]
english_sentences = [english[i] for i in valid_sentence_indicies]

# %%
import torch

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
eng_vocab_size = len(english_vocab)

transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          eng_vocab_size,
                          pidgin_to_index,
                          english_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)

transformer

# %%
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, pidgin_sentences, english_sentences):
        self.english_sentences = english_sentences
        self.pidgin_sentences = pidgin_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.pidgin_sentences[idx], self.english_sentences[idx]
        
# %%
dataset = TextDataset(pidgin_sentences, english_sentences)
dataset[1]

# %%
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break
        
# %%
from torch import nn

criterion = nn.CrossEntropyLoss(ignore_index=pidgin_to_index[PADDING_TOKEN],
                                reduction='none')

for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
NEG_INFTY = -1e9

def create_masks(pdg_batch, eng_batch):
    num_sentences = len(pdg_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, pdg_sentence_length = len(eng_batch[idx]), len(pdg_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      pdg_chars_to_padding_mask = np.arange(pdg_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, pdg_chars_to_padding_mask] = True
      encoder_padding_mask[idx, pdg_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, pdg_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, eng_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask
    
# %%
transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        pdg_batch, eng_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(pdg_batch, eng_batch)
        optim.zero_grad()
        eng_predictions = transformer(pdg_batch,
                                        eng_batch,
                                        encoder_self_attention_mask.to(device),
                                        decoder_self_attention_mask.to(device),
                                        decoder_cross_attention_mask.to(device),
                                        enc_start_token=False,
                                        enc_end_token=False,
                                        dec_start_token=True,
                                        dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(eng_batch, start_token=False, end_token=True)
        loss = criterion(
            eng_predictions.view(-1, eng_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == english_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"Pidgin: {pdg_batch[0]}")
            print(f"English Translation: {eng_batch[0]}")
            english_sentence_predicted = torch.argmax(eng_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in english_sentence_predicted:
                if idx == english_to_index[END_TOKEN]:
                break
                predicted_sentence += index_to_english[idx.item()]
            print(f"English Prediction: {predicted_sentence}")


            transformer.eval()
            eng_sentence = ("",)
            pdg_sentence = (f"{pdg_batch[0]}",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(pdg_sentence, eng_sentence)
                predictions = transformer(pdg_sentence,
                                            eng_sentence,
                                            encoder_self_attention_mask.to(device),
                                            decoder_self_attention_mask.to(device),
                                            decoder_cross_attention_mask.to(device),
                                            enc_start_token=False,
                                            enc_end_token=False,
                                            dec_start_token=True,
                                            dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter]
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_english[next_token_index]
                eng_sentence = (eng_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                    break

            print(f"Evaluation translation ({pdg_batch[0]}) : {eng_sentence}")
            print("-------------------------------------------")
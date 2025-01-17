### download data

# load the label and caption data from google drive
import pandas as pd
from google.colab import drive
drive.mount("/content/drive")
train_data = pd.read_csv("/content/drive/MyDrive/train.csv")
val_data = pd.read_csv("/content/drive/MyDrive/val.csv")
test_data = pd.read_csv("/content/drive/MyDrive/CONDA_test_original.csv")

print(test_data.shape)
test_data = test_data[["utterance","slotClasses"]]
test_data.columns = ["sents", "labels"]
test_data = test_data.dropna(axis=0, how="any")
print(test_data.shape)
test_data.head()

### data pre-processing

# tokenization
import re
import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize, TweetTokenizer
tweet_tokenizer = TweetTokenizer()

# define functions to implement data pre-processing
def clean_str(x):
    x = re.sub(r'[!#\[\]]', "", x.lower())
    x = re.sub(r':+', ":", x)
    x = re.sub(r':d+', ":d", x)
    x = re.sub(r'\d\d:\d\d|\d:\d\d|\d\d:\d|\d:\d', ":d", x)
    x = re.sub(r':\d|:\D', ":d", x)
    x = re.sub(r'=+', "=", x)
    x = re.sub(r'=\d=|=\D=|=\d|=\D', ":d", x)
    x = re.sub(r';_;|;-;|;_:|;\d|;\D', ":d", x)
    return x

# if number of words and labels are not matched, drop out
def clean_data(x_data, y_data):
  x = []
  y = []
  for i in range(len(x_data)):
    if len(x_data[i]) == len(y_data[i]):
      x.append(x_data[i])
      y.append(y_data[i])
  return x, y

# clean data, word tokenization
# ==== train ====
x_train = []
for i in train_data["sents"]:
  x_train.append(tweet_tokenizer.tokenize(clean_str(i)))
y_train = []
for i in train_data["labels"]:
  y_train.append(word_tokenize(i))

x_train, y_train = clean_data(x_train, y_train)
print(len(x_train), x_train[:10])
print(len(y_train), y_train[:10])
print()

# ==== validation ====
x_val = []
for i in val_data["sents"]:
  x_val.append(tweet_tokenizer.tokenize(clean_str(i)))
y_val = []
for i in val_data["labels"]:
  y_val.append(word_tokenize(i))

x_val, y_val = clean_data(x_val, y_val)
print(len(x_val), x_val[:10])
print(len(y_val), y_val[:10])
print()

# ==== test ====
x_test = []
for i in test_data["sents"]:
  x_test.append(tweet_tokenizer.tokenize(clean_str(i)))
y_test = []
for i in test_data["labels"]:
  y_test.append(word_tokenize(i))

x_test, y_test = clean_data(x_test, y_test)
print(len(x_test), x_test[:10])
print(len(y_test), y_test[:10])

"""
# check if the number of words meets the requirement
count = 0
for i in x_test:
  count = count + len(i)
print("number of words in x_test:", count)
"""

# generate word_to_ix and tag_to_ix
word_to_ix = {}
for sentence in x_train + x_val + x_test:
    for word in sentence:
        word = word.lower()
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
word_list = list(word_to_ix.keys())
ix_to_word = {idx: w for idx, w in enumerate(word_list)}
print(len(word_list), word_list[:10])
# print(len(word_to_ix), word_to_ix)
# print(len(ix_to_word), ix_to_word)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {START_TAG:0, STOP_TAG:1}
for tags in y_train + y_val + y_test:
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
print(len(tag_to_ix), tag_to_ix)

# generate embedding matrix
# self-trained word embedding model
sentences = np.concatenate((x_train, x_val, x_test), 0)

from gensim.models import Word2Vec, FastText
w2v_model = FastText(sentences=sentences, size=50, window=1, min_count=0, workers=2, sg=0)

# word embeddings
import warnings
warnings.filterwarnings("ignore")

w2v_embed_matrix = []
for i, word in enumerate(word_list):
    if word in w2v_model:
        w2v_embed_matrix.append(w2v_model[word])
    else:
        w2v_embed_matrix.append([0] * w2v_model.vector_size)

w2v_embed_matrix = np.array(w2v_embed_matrix)
print("w2v_embed_matrix:", w2v_embed_matrix.shape)

# convert data into indexes
def to_index(data, to_ix):
    input_index_list = []
    for sent in data:
        input_index_list.append([to_ix[w] for w in sent])
    return input_index_list

train_input_index = to_index(x_train, word_to_ix)
train_output_index = to_index(y_train, tag_to_ix)
val_input_index = to_index(x_val, word_to_ix)
val_output_index = to_index(y_val, tag_to_ix)
test_input_index = to_index(x_test, word_to_ix)
test_output_index = to_index(y_test, tag_to_ix)

# print(train_input_index[:10])
# print(train_output_index[:10])
# print(val_input_index[:10])
# print(val_output_index[:10])
# print(test_input_index[:10])
# print(test_output_index[:10])

### proposed model: BRAR

# label forcing
import torch
import torch.nn.functional as F

def get_lf_prob(x_train_list, y_train_list):
  word_label_cnt = []
  for ix in range(len(word_list)):
    cnt = {}
    for i in range(len(x_train_list)):
      if x_train_list[i] == ix:
        cnt[y_train_list[i]] = cnt.get(y_train_list[i], 0) + 1
    cnt = sorted(cnt.items(), key = lambda x:x[1], reverse=True)
    word_label_cnt.append(cnt)

  lf_prob = torch.zeros(len(word_label_cnt), len(tag_to_ix))
  for i in range(len(word_label_cnt)):
    for j in word_label_cnt[i]:
      lf_prob[i][j[0]] = j[1]
  lf_prob = F.softmax(lf_prob, dim=1)
  return lf_prob

"""
# get each element from training set
x_train_list = []
for i in train_input_index:
  for j in i:
    x_train_list.append(j)

y_train_list = []
for i in train_output_index:
  for j in i:
    y_train_list.append(j)

print("train words:", len(x_train_list), len(y_train_list))
lf_prob_train = get_lf_prob(x_train_list, y_train_list)
print(lf_prob_train.size())
"""

# get each element from training set + validation set
x_train_list = []
for i in train_input_index + val_input_index:
  for j in i:
    x_train_list.append(j)

y_train_list = []
for i in train_output_index + val_output_index:
  for j in i:
    y_train_list.append(j)

print("train + val words:", len(x_train_list), len(y_train_list))
lf_prob_train_val = get_lf_prob(x_train_list, y_train_list)
print(lf_prob_train_val.size())

# model initialization
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
torch.manual_seed(1)

############################ Function Defination ############################

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# evaluation metrics
def cal_acc(model, input_index, output_index):

    predictions = []
    for i in range(len(input_index)):
        torch_input = torch.tensor(input_index[i], dtype=torch.long).to(device)
        torch_pred = model(torch_input)

        for j in torch_pred:
            predictions.append(j)

    ground_truth = []
    for i in output_index:
        for j in i:
            ground_truth.append(j)

    count = 0
    for i in range(len(predictions)):
        if predictions[i] == ground_truth[i]:
            count += 1
    accuracy = count / len(predictions)
    return predictions, ground_truth, accuracy


# decode index to labels
def decode_output(output_list):
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    return [ix_to_tag[output] for output in output_list]


############################ Model Defination ############################


class BRAR(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, embedding_matrix, hidden_dim, n_layers, attn_method, lf_prob):
        super(BRAR, self).__init__()
        self.vocab_size = vocab_size # embedding
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.embedding_dim = embedding_dim # BiLSTM
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attn_method = attn_method # attention
        self.lf_prob = lf_prob.to(device) # LF

        # use the embedding matrix as the initial weights of nn.Embedding
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_matrix))

        # define BiLSTM
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=n_layers, bidirectional=True)

        # intialise weights of the attention mechanism
        self.attn_weight = nn.Parameter(torch.zeros(1)).to(device)
        self.general_weights = nn.Parameter(torch.randn(1, self.hidden_dim, self.hidden_dim)).to(device)
        self.location_weights = nn.Parameter(torch.randn(1, self.hidden_dim, 1)).to(device)

        # define fully-connected layer
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()

        # never transfer to the start tag and we never transfer from the stop tag
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000


    def init_hidden(self):
        return (torch.randn(2*self.n_layers, 1, self.hidden_dim // 2).to(device),
                torch.randn(2*self.n_layers, 1, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        # forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):

                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def attention(self, lstm_out, final_state):

        if self.attn_method == "dot_product":

          # final_hidden = (batch=1, hidden_size*2=10, 1)
          final_hidden = final_state.view(-1, self.hidden_dim, 1)

          # attn_weights = (1,8,10)*(1,10,1) -> (1,8,1) -> seueeze(2) -> (batch=1, seq_len=8)
          attn_weights = torch.bmm(lstm_out, final_hidden).squeeze(2)
          # soft_attn_weights = (1,8)
          soft_attn_weights = F.softmax(attn_weights, 1)

          # attn_out = (1,10,8)*(1,8,1) -> (1,10,1) -> sequeeze(2) -> (batch=1, hidden_size*2=10)
          attn_out = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        elif self.attn_method == "scaled_dot_product":

          final_hidden = final_state.view(-1, self.hidden_dim, 1)
          attn_weights = (torch.bmm(lstm_out, final_hidden) / np.sqrt(self.hidden_dim)).squeeze(2)
          soft_attn_weights = F.softmax(attn_weights, 1)
          attn_out = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        elif self.attn_method == "general":

          final_hidden = final_state.view(-1, self.hidden_dim, 1)
          attn_weights = torch.bmm(torch.bmm(lstm_out, self.general_weights), final_hidden).squeeze(2)
          soft_attn_weights = F.softmax(attn_weights, 1)
          attn_out = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        elif self.attn_method == "location_based":

          final_hidden = final_state.view(-1, self.hidden_dim, 1)
          attn_weights = torch.bmm(lstm_out, self.location_weights).squeeze(2)
          soft_attn_weights = F.softmax(attn_weights, 1)
          attn_out = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        else:
          attn_out = torch.zeros(1).to(device)
        return attn_out

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()

        # embeds = (seq_len=8, batch=1, embedding_dim=50)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)

        # lstm_out = (seq_len=8, batch=1, hidden_size*2=10)
        # h_n, c_n = (n_layer=2, batch=1, hidden_size=5)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # lstm_out = (batch=1, seq_len=8, hidden_size*2=10)
        lstm_out = lstm_out.permute(1, 0, 2)

        # attn_out = (batch=1, hidden_size*2=10)
        attn_out = self.attention(lstm_out, self.hidden[0])

        # lstm_out.shape = (seq_len=8, hidden_size*2=10)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        # res_out.shape = (seq_len=8, hidden_size*2=10) ![important] take attn_out as residual
        res_out = lstm_out + attn_out * self.attn_weight

        # output.shape = (seq_len=8, tagset_size=9)
        output = self.hidden2tag(res_out)
        return output

    def _score_sentence(self, feats, tags):
        # gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):

                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # transition to stop tag
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # pop off the start tag, we dont want to return that to the caller
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG] # sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # get emission probability
        emi_prob = self._get_lstm_features(sentence)
        emi_prob = emi_prob + self.lf_prob[sentence].view(-1, self.tagset_size) # ![important] label forcing

        # find the best path, given the features
        score, tag_seq = self._viterbi_decode(emi_prob)
        return tag_seq


############################ Create A New Model ############################

# hyper-parameters
EMBEDDING_DIM = w2v_embed_matrix.shape[1]
EMBEDDINGS = w2v_embed_matrix
HIDDEN_DIM = 10
N_LAYERS = 1
ATTN_METHOD = "general"
LF_PROB = lf_prob_train_val
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BRAR(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, EMBEDDINGS, HIDDEN_DIM, N_LAYERS, ATTN_METHOD, LF_PROB).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4) # SGD with lr of 0.1 is better


############################ Model Training Part ############################


import datetime
for epoch in range(2):
    time1 = datetime.datetime.now()
    train_loss = 0

    model.train()
    for i, idxs in enumerate((train_input_index + val_input_index)):
        tags_index = (train_output_index + val_output_index)[i]
        sentence_in = torch.tensor(idxs, dtype=torch.long).to(device)
        targets = torch.tensor(tags_index, dtype=torch.long).to(device)

        model.zero_grad()
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    # call the cal_acc functions
    _, _, train_acc = cal_acc(model, train_input_index, train_output_index)
    _, _, val_acc = cal_acc(model, val_input_index, val_output_index)

    time2 = datetime.datetime.now()

    print("epoch: {:} // train acc: {:.2f}% // val acc: {:.2f}% // time: {:.2f}s.".
          format(epoch+1, train_acc*100, val_acc*100, (time2-time1).total_seconds()))

from sklearn.metrics import f1_score, classification_report

test_y_pred, test_y_truth, _ = cal_acc(model, test_input_index, test_output_index)

# calcualte overall F1 without label "O"
y_truth_new = np.array([test_y_truth[i] for i in range(len(test_y_pred)) if (test_y_pred[i] != 2) & (test_y_truth[i] != 2)])
y_pred_new = np.array([test_y_pred[i] for i in range(len(test_y_pred)) if (test_y_pred[i] != 2) & (test_y_truth[i] != 2)])
print("overall f1-score ", round(f1_score(y_truth_new, y_pred_new, average="micro"), 4))

print("-"*54)

# calculate F1 scores for T P S D C O
target_names = ["T", "P", "SEPA", "S", "D", "C", "O"]
print(classification_report(np.array(test_y_truth), np.array(test_y_pred), labels=[3, 4, 5, 6, 7, 8, 2], target_names=target_names, digits=4))
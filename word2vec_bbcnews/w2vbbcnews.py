#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F


# In[2]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')


# In[3]:


print(type(STOPWORDS))


# In[4]:


import re, string 
import pandas as pd   
from collections import defaultdict
import spacy
from sklearn.manifold import TSNE

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('bbc_data.csv')
df.columns= ['news_article'] 
def clean_text(text):
    '''Make text lowercase, remove square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub(r'[\[\]\(\)\{\}]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    
    text = word_tokenize(text)
    # Remove a sentence if it is only one word long
    if len(text) > 2:
        return [word for word in text if word not in STOPWORDS]

tokenized_corpus = df['news_article'].apply(lambda x: clean_text(x))
print(tokenized_corpus)


# In[5]:


print(tokenized_corpus[0])
print(type(tokenized_corpus[0][0]))


# In[6]:


vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)


# In[7]:


vocabulary_size


# In[8]:


word2idx


# In[9]:


window_size = 3
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array


# In[10]:


idx_pairs[:10]


# In[9]:


def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x
  
#Input layer is just the center word encoded in one-hot manner. It dimensions are [1, vocabulary_size]


# In[10]:


embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')


# Word2Vec

# In[11]:


df_clean = pd.DataFrame(tokenized_corpus)
df_clean


# In[17]:


nlp = spacy.load('en')
df_clean = pd.DataFrame(tokenized_corpus)
def lemmatizer(text):        
    sent = []
    doc = nlp(" ".join(text))
    for word in doc:
        sent.append(word.lemma_)
    return sent

df_clean["text_lemmatize"] =  df_clean.apply(lambda x: lemmatizer(x['news_article']), axis=1)


# In[18]:


df_clean


# In[19]:


sentences = [row for row in df_clean['text_lemmatize']]
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
        
sorted(word_freq, key=word_freq.get, reverse=True)[:10]


# In[20]:


import sys
w2v_model = Word2Vec(min_count=200,
                     window=5,
                     size=100,
                     workers=4)
                     
w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
w2v_model.init_sims(replace=True)


# In[21]:


w2v_model.wv.most_similar(positive=['people'])


# In[22]:


w2v_model.wv.similarity('company', 'business')


# In[30]:


voc = list(w2v_model.wv.vocab)
print(len(voc))
print(voc)


# In[ ]:





# In[24]:


w2v_model.wv.most_similar(positive=['US'])


# In[31]:


sentences_full = [row for row in df_clean['news_article']]
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
        
sorted(word_freq, key=word_freq.get, reverse=True)[:10]


# In[32]:


import sys
w2v_model = Word2Vec(min_count=10,
                     window=5,
                     size=500,
                     workers=4)
                     
w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
w2v_model.init_sims(replace=True)


# In[33]:


voc = list(w2v_model.wv.vocab)
print(len(voc))
print(voc)


# In[34]:


w2v_model.wv.most_similar(positive=['people'])


# In[35]:


w2v_model.wv.similarity('company', 'business')


# In[36]:


w2v_model.wv.most_similar(positive=['america'])


# In[37]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

def tsne_plot(model):
    "Create TSNE model and plot it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
   
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    # print(x, y)        
    plt.figure(figsize=(25,25)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    
tsne_plot(w2v_model)


# In[40]:


vector = w2v_model.wv['america']  # numpy vector of a word
print(len(vector))
print(vector)


# In[ ]:





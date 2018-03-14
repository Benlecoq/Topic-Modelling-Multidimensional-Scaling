
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Load data
data = fetch_20newsgroups() 
df = pd.DataFrame({'target': data.target , 'text': data.data})

print(df['text'].iloc[1])


# In[69]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Data Preprocessing (single function from "Fine 20newsgroups Dataset Preprocessing" steps)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(text):
    dirt = text.split("\n\n")[0] # locate headers before first double new line
    stripheaders = text.replace(dirt, '') # Remove headers
    stripline = "\n".join([line for line in stripheaders.splitlines() if not "@" in line]) # Remove line containing email addresses
    char = ''.join(ch for ch in stripline if ch not in exclude) # Remove punctuation
    stopwords = " ".join([i for i in char.lower().split() if i not in stop]) # Remove stop words
    lemmatized = " ".join(lemma.lemmatize(word) for word in stopwords.split()) # Lemmatize
    return lemmatized

clean_doc = [clean(doc).split() for doc in df.text]   

print(clean_doc[1])


# In[72]:


import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis.gensim

# Create dictionary and corpus
dictionary = corpora.Dictionary(clean_doc)
corpus = [dictionary.doc2bow(doc) for doc in clean_doc]

# Train model
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

# Graph topics via Multidimensional Scaling
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda, corpus, dictionary)


# In[4]:





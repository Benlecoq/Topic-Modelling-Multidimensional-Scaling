
# coding: utf-8

# In[34]:


from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# Load dataset
data = fetch_20newsgroups() 
df = pd.DataFrame({'text': data.data})

# Isolate one text to display
text = df.text.iloc[1]

print(text)


# In[35]:


# Strip first block corresponding to email details

def stripheaders(text):
    dirt = text.split("\n\n")[0] # locate headers as everything before first double new line and save as string
    strip = text.replace(dirt, '') # Strip that string from full text
    return strip

stepbystep = stripheaders(text)

print(stepbystep)


# In[36]:


# Strip lines containing "@" (usually emails)

def stripcontains (text):
    strip = "\n".join([line for line in text.splitlines() if not "@" in line])
    return strip

stepbystep = stripcontains(stepbystep)

print(stepbystep)


# In[37]:


import string

# Lowercase and remove punctuation 

exclude = set(string.punctuation) 

def low_punc (text): 
    nopunc = ''.join(ch for ch in text.lower() if ch not in exclude)
    return nopunc

stepbystep = low_punc(stepbystep)

print(stepbystep)


# In[38]:


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords 

# Remove Stopwords and Lemmatize

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def stops_lemma (text):
    stops = " ".join([i for i in text.split() if i not in stop]) 
    lemmatized = " ".join(lemma.lemmatize(word) for word in stops.split()) 
    return lemmatized

stepbystep = stops_lemma(stepbystep)

print(stepbystep)


# In[39]:


# Data Preprocessing all in one

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

# Apply to all texts
clean_doc = [clean(doc).split() for doc in df.text]   

print(clean_doc[1])


# In[40]:


import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import pyLDAvis.gensim

# Create dictionary and corpus
dictionary = corpora.Dictionary(clean_doc)
corpus = [dictionary.doc2bow(doc) for doc in clean_doc]

# Train model
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

#Original dataset topics
print(data.target_names)

# Graph topics via Multidimensional Scaling (Relevence metric optimal around λ = 0.3 for topic interpretability)
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda, corpus, dictionary)


#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
#for visualization
import matplotlib.pyplot as plt
#forregular expressions
import re
#for handling string
import string
#for mathmatical operations
import math
import seaborn as sns
import time

#Loading data(Read data as dataframe)
df = pd.read_csv("C:\\Users\\AcerPC\\Desktop\\project\\datasets\\DATASETS\\electronics_prod.csv", low_memory=False)
df

#for checking rows and columns of the dataset
#print("shape of data",df.shape)


# In[2]:


df.describe()


# In[3]:


df.info()  #reviews.text category has minimum missing data (34659/34660) -> Good news!


# In[4]:


df["productId"].unique()
df["productId"].nunique()


# In[5]:


productId_unique = len(df["productId"].unique())
print("Number of Unique product ids: " + str(productId_unique))


# In[6]:


df["reviews_rating"].count()


# In[7]:


sns.countplot(x = 'reviews_rating', data = df)


# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit
print("Before {}".format(len(df)))
dfafter = df.dropna(subset=["reviews_rating"]) # removes all NAN in reviews.rating
print("After {}".format(len(dfafter)))
dfafter["reviews_rating"] = dfafter["reviews_rating"].astype(int)


# In[ ]:





# In[9]:


split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
for train_index, test_index in split.split(dfafter, dfafter["reviews_rating"]): 
    strat_train = dfafter.reindex(train_index)
    strat_test = dfafter.reindex(test_index)


# In[10]:


len(strat_train)


# In[11]:


strat_train["reviews_rating"].value_counts() # value_count() counts all the values based on column


# In[12]:


len(strat_test)


# In[13]:


strat_test["reviews_rating"].value_counts()#/len(strat_test)


# In[15]:


reviews = strat_train.copy()
reviews.head()


# In[16]:


len(reviews["name"].unique()), len(reviews["productId"].unique())


# In[17]:


reviews.info()


# In[18]:


reviews.groupby("productId")["name"].unique()


# In[19]:


different_names = reviews[reviews["productId"] == "B00L9EPT8O,B01E6AO69U"]["name"].unique()
for name in different_names:
    print(name)


# In[20]:


reviews[reviews["productId"] == "B00L9EPT8O,B01E6AO69U"]["name"].value_counts()


# In[21]:


fig = plt.figure(figsize=(16,10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
reviews["productId"].value_counts().plot(kind="bar", ax=ax1, title="product Id Frequency")
np.log10(reviews["productId"].value_counts()).plot(kind="bar", ax=ax2, title="product Id Frequency (Log10 Adjusted)") 
plt.show()


# In[22]:


reviews["reviews_rating"].mean()


# In[23]:


productId_count_ix = reviews["productId"].value_counts().index
plt.subplots(2,1,figsize=(16,12))
plt.subplot(2,1,1)
reviews["productId"].value_counts().plot(kind="bar", title="product Id Frequency")
plt.subplot(2,1,2)
sns.pointplot(x="productId", y="reviews_rating", order=productId_count_ix, data=reviews)
plt.xticks(rotation=90)
plt.show()


# In[24]:


corr_matrix = reviews.corr()
corr_matrix


# In[26]:


counts = reviews["productId"].value_counts().to_frame()
counts.head(5)


# In[27]:


avg_rating = reviews.groupby("productId")["reviews_rating"].mean().to_frame()
avg_rating.head()


# In[28]:


table = counts.join(avg_rating)
table.head(30)


# In[29]:


plt.scatter("productId", "reviews_rating", data=table)
table.corr()


# In[30]:


def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif (rating == 2) or (rating == 1):
        return "Negative"
# Add sentiments to the data
strat_train["Sentiment"] = strat_train["reviews_rating"].apply(sentiments)
strat_test["Sentiment"] = strat_test["reviews_rating"].apply(sentiments)
strat_train["Sentiment"][:20]


# In[31]:


X_train = strat_train["reviews_text"]
X_train_targetSentiment = strat_train["Sentiment"]
X_test = strat_test["reviews_text"]
X_test_targetSentiment = strat_test["Sentiment"]
print(len(X_train), len(X_test))


# In[32]:


#Replace "nan" with space
X_train = X_train.fillna(' ')
X_test = X_test.fillna(' ')
X_train_targetSentiment = X_train_targetSentiment.fillna(' ')
X_test_targetSentiment = X_test_targetSentiment.fillna(' ')


# In[34]:


df = df[['reviews_rating' , 'reviews_text' , 'reviews_title' , 'reviews_username']]


# In[35]:


print(df.isnull().sum()) #Checking for null values


# In[36]:


df["pos/neg"] = df["reviews_rating"]>=4
df["pos/neg"] = df["pos/neg"].replace([True , False] , ["pos" , "neg"])


# In[37]:


df.head()


# In[38]:


sns.countplot(df['pos/neg'], data = df)


# In[39]:


cleanup_re = re.compile('[^a-z]+')
def clean_up(review):
    review = str(review)
    review = review.lower()
    review = cleanup_re.sub(' ', review).strip()
    #sentence = " ".join(nltk.word_tokenize(sentence))
    return review

df["Clean"] = df["reviews_text"].apply(clean_up)
df["Cleaned_name"] = df["name"].apply(clean_up)
#null["Clean"] = null["reviews_text"].apply(clean_up)


# In[40]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
#import nltk.classify.util
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn import metrics
#from sklearn.metrics import roc_curve, auc
#from nltk.classify import NaiveBayesClassifier
#import numpy as np
#import re
#import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#from nltk.stem.snowball import SnowballStemmer as ss
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer


# In[41]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)


matplotlib.rcParams['font.size']=12          
matplotlib.rcParams['savefig.dpi']=100             
matplotlib.rcParams['figure.subplot.bottom']=.1 


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=300,
        max_font_size=40, 
        scale=3,
        random_state=1
        
    ).generate(str(data))
    
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
show_wordcloud(df["Clean"])


# In[42]:


show_wordcloud(df["Clean"][df['pos/neg'] == "pos"] , title="Positive Words")


# In[43]:


show_wordcloud(df["Clean"][df['pos/neg'] == "neg"] , title="Negative words")


# In[48]:


from textblob import TextBlob
import pandas as pd
import nltk

cleanup_re = re.compile('[^a-z]+')
def clean_up(review):
    review = str(review)
    review = review.lower()
    review = cleanup_re.sub(' ', review).strip()
    #sentence = " ".join(nltk.word_tokenize(sentence))
    return review

df["Cleaned_reviews"] = df["reviews_text"].apply(clean_up)
df["Cleaned_name"] = df["name"].apply(clean_up)

df['Polarity'] = df.Cleaned_reviews.apply(lambda x: np.mean(
              [TextBlob(r[0]).sentiment.polarity for r in TextBlob(x).ngrams(1)]))


# In[51]:


df


# In[49]:


dfNEG=df[df.reviews_rating<3]
dfNEG


# In[50]:


dfNEG[dfNEG.Polarity<=-0.1].head(20) # showing the badest review from all reviews


# In[ ]:





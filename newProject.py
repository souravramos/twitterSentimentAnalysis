import time
import pandas as pd
import re
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

start_time = time.time()
train = pd.read_csv("train_E6oV3lV.csv")
test = pd.read_csv("test_tweets_anuFYb8.csv")

combi = train.append(test, ignore_index=True)
pd.set_option('display.max_rows',80)
pd.set_option('display.max_columns',80)
#print(combi.head())

# user defined function to remove unwanted text patterns.
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, str(input_txt))
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# remove twitter handles from tweets (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], '@[\w]*')

# remove special characters, numbers and punctuations from tweets
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace('[^a-zA-Z#]', ' ')

# remove short words ( words having wordlength less than 3 )
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x : ' '.join([w for w in x.split()
                                                                    if len(w)>3]))
# tokenizing all clean tweets into tokens (words) (list of tokens)
tokenized_tweet = combi['tidy_tweet'].apply(lambda x : x.split())

# stemming
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x : [stemmer.stem(i) for i in x])

# stitching tokens back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet
print(combi.head())

# Understanding common words used in our dataset : WordCloud
all_words = ' '.join([text for text in combi['tidy_tweet']])
wordcloud = WordCloud(width=800, height=800, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
#plt.show()

# Words in non racist/sexist tweets
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label']==0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('on')
plt.show()
           
# Words in racist/sexist tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label']==1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)   
    return hashtags

# extracting hashtags from non-racist/sexist tweets
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label']==0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label']==1])

# unnesting list
HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])

# Non-Racist/Sexist Tweets
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag' : list(a.keys()),
                  'Count' : list(a.values())})

# selecting top 10 most frequent hashtags
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=d, x='Hashtag', y='Count')
ax.set(ylabel='Count')
plt.show()

# Racist/Sexist Tweets
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag' : list(b.keys()),
                  'Count' : list(b.values())})

# selecting top 10 most frequent hashtags
e = e.nlargest(columns='Count', n=10)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=e, x='Hashtag', y='Count')
ax.set(ylabel='Count')
plt.show()

# Bag-of-Words Features
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000,
                                 stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

# Building model using Bag-of-Words features
train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],
                                                          random_state=42, test_size=0.3)

# Logistic Regression
lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)   # training the data

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 then 1 else 0
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int)) # calculating f1 score 

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:, 1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id', 'label']]
submission.to_csv('sample_submission_gfvA5FD.csv', index=False) # writing data to a csv file 

gnb = GaussianNB()
gnb.fit(xtrain_bow.toarray(), ytrain)     # training the data
prediction = gnb.predict_proba(xvalid_bow.toarray())
prediction_int = prediction[:, 1] >= 0.3 # if prediction is greater than or equal to 0.3 then 1 else 0
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int)) # calculating f1 score

#filename = 'finalized_model.sav'
#pickle.dump(lreg, open(filename, 'wb'))

print('---- %s seconds ----' % (time.time()-start_time))
                                    
           

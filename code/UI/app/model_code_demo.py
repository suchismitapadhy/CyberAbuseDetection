
# coding: utf-8

# ### Imports

# In[1]:

import pandas as pd
import nltk, re, string
import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import preprocessing
from sklearn import svm
from nltk.corpus import stopwords

from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import itertools
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re, collections
from nltk.corpus import words as w
import pickle
from nltk.corpus import *
from nltk.collocations import *
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


# ### Preprocess Data

# Read the csv file and make a dataframe.
# - For training: Randomize and Divide it into 80:20 partitions

# In[2]:

def load_Dataset(run="train"):
    df = pd.read_csv("train.csv")
    df = df[df["Comment"].notnull()]
    df.apply(np.random.permutation)
    if run=="train":
        df_train = df[:round(0.8*len(df))]
        df_test = df[round(0.8*len(df)):]
    elif run=="test":
        df_train = df
        df_test = pd.read_csv("test_with_solutions.csv")
    elif run=="test1":
        df_train = df
        df_test = pd.read_csv("impermium_verification_labels.csv")
        df_test.describe()
        #del(df_test['ID'])
    return df_train, df_test


# In[3]:

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1   
    return model

with open("big.txt", "r") as big:
    word_corpus = big.read()
for word in w.words():
    word_corpus += word
    

NWORDS = train(words(word_corpus))
with open("list_of_abuses.txt", "r") as abuse_list:
    abuses = abuse_list.read().split()
    for abuse in abuses:
        NWORDS[abuse] = 100

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
#     print(word)
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words):
    try:
        return [int(w) for w in words] #to take care of purely numeric words
    except:
        return set(w for w in words if w.lower() in NWORDS)

def correct(word):
    if word[0] not in alphabet: 
        return word
    else:
        word = re.sub(r'(.)\1+', r'\1\1', word)
        candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
        return max(candidates, key=NWORDS.get)


# In[4]:

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w!@#$%^&*]+)', # To group symbols together
    r'(?:[\w_]+)', # other words
    
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    tokens = tokens_re.findall(s)
    for i in range(len(tokens)):
        clean_token = correct(tokens[i])
        tokens[i] = clean_token
    return tokens
 
def preprocess(word, lowercase=False):
    tokens = tokenize(word)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


# ### Tokenization

# This function takes a text and does the following to return the tokens:
# * Use nltk's TweetTokenizer to get tokens
# * Use wordNetLemmatizer for lemmatization
# * Use porterStemmer to stem the resulting tokens

# In[5]:

def build_tokens(text):
    tweetTokenizer = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tweetTokenizer.tokenize(text)
    #tokens = preprocess(text, lowercase=True)
    #tokens = [nltk.WordNetLemmatizer().lemmatize(token) for token in tokens]
    #tokens= [nltk.PorterStemmer().stem(token) for token in tokens]
    return tokens


# Lemmatizing, Stemming and custom preprocessing reduced the accuracy. Tweettokenizer worked better.

# ## Pipeline and Adding Custom features

# #### Adding additional features

# Few custom features are added - percentage of bad words in a sentence as listed in the bad words file and the compound, negative and positive values from vader sentiment analysis and number of words in comments.

# In[6]:

class AdditionalFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def get_feature_names(self):
        return (['percent_bad','vader_compound','num_words','vader_neg','vader_pos','ur_bad','word2vec'])
    
    def num_bad(self, df):
        #get number of words in each sentence
        with open("list_of_abuses.txt", "r") as abuse_list:
             abuses = abuse_list.read().split("\n")
        num_words = [len(word) for word in df]
        
        #get percent of abusive words in each sentence
        num_abuses = 0
        for abuse in abuses:
            num_abuses += 1
        # number of badwords in list of abuses
        num_bad = [np.sum([word.lower().count(abuse) for abuse in abuses])
                                            for word in df]
        norm_bad = np.array(num_bad) / np.array(num_words, dtype=np.float)
        return norm_bad

    def ur_bad(self, df):
        #look for you are "bad words" in text
        with open("list_of_abuses.txt", "r") as abuse_list:
             abuses = abuse_list.read().split("\n")
        ur_bad_bool = []
        for sentence in df:
            abuse_found = False
            for words in abuses:
                abuse = r'(\b%s\b)' %words
                if (re.findall(r"\byou are\b",sentence.lower()) != []) and (re.findall(abuse,sentence.lower()) != []):
                    if (re.findall(r"\bnot\b",sentence.lower()) == []):
                        ur_bad_bool.append(True)
                        abuse_found = True
                        break
                elif (re.findall(r"\bur\b",sentence.lower()) != []) and (re.findall(abuse,sentence.lower()) != []):
                    if (re.findall(r"\bnot\b",sentence.lower()) == []):
                        ur_bad_bool.append(True)
                        abuse_found = True
                        break
                elif (re.findall(r"\byour\b",sentence.lower()) != []) and (re.findall(abuse,sentence.lower()) != []):
                    if (re.findall(r"\bnot\b",sentence.lower()) == []):
                        ur_bad_bool.append(True)
                        abuse_found = True
                        break
                elif (re.findall(r"\byou're\b",sentence.lower()) != []) and (re.findall(abuse,sentence.lower()) != []):
                    if (re.findall(r"\bnot\b",sentence.lower()) == []):
                        ur_bad_bool.append(True)
                        abuse_found = True
                        break
                elif (re.findall(r"\bu r\b",sentence.lower()) != []) and (re.findall(abuse,sentence.lower()) != []):
                    if (re.findall(r"\bnot\b",sentence.lower()) == []):
                        ur_bad_bool.append(True)
                        abuse_found = True
                        break
            if abuse_found == False:
                ur_bad_bool.append(False)            
        return ur_bad_bool
    
    def num_words(self, df):
        #get number of words in each sentence
        num_words = [len(word) for word in df]
        return num_words
    
    def vader_helper(self, df):
        #vader analysis
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['compound'])
        return vader_feature
    
    def vader_helper_neg(self, df):
        #vader analysis
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['neg'])
        return vader_feature
    
    def vader_helper_neu(self, df):
        #vader analysis
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['neu'])
        return vader_feature
    
    def vader_helper_pos(self, df):
        #vader analysis
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['pos'])
        return vader_feature
    
    def transform(self, df, y=None):     
        #add both the features to an array
        X = np.array([self.vader_helper(df),self.vader_helper_pos(df),self.vader_helper_neg(df),self.vader_helper_neu(df),self.ur_bad(df)]).T
        #X = np.array([self.num_bad(df),self.vader_helper(df)]).T
        #X.reshape(-1, 1) #use if only 1 feature
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)      

    def fit(self, df, y=None):
        return self


# In[7]:

class Word2VecFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def get_feature_names(self):
        return (['word2vec'])
    
    def make_tweetVector(self, df_comments):
        learned_word_vectors,learned_dictionary = pickle.load( open( "glove_twitter_200d.pkl", "rb" ) ) 
        tweetVector = np.zeros((len(df_comments),200),dtype='float')
        count = 0 
        for text in df_comments:
            valids= 0
            tokens = build_tokens(text)
            rowVector = np.zeros(200,)
            for token in tokens:
                if token not in learned_dictionary:
                    continue
                else:
                    valids+=1
                    vec = learned_word_vectors[learned_dictionary[token.lower()]]
                    rowVector = np.add(rowVector,vec)
            tweetVector[count] = rowVector/valids
            count+=1
        return tweetVector
    
    def transform(self, df, y=None):     
        #add both the features to an array
        X = np.array(self.make_tweetVector(df))
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)      

    def fit(self, df, y=None):
        return self


# The custom features are stacked along with the features got from TF-IDF char and word analyzer

# In[8]:

stopwords = ['a','an','the']
def all_features():
    features = []
    custom_features = AdditionalFeatureExtractor() # this class includes my custom features 
    
    vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,6), analyzer= "char", stop_words = stopwords, tokenizer= build_tokens)
    vect1 = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,6), analyzer= "word", stop_words = stopwords, tokenizer= build_tokens)
    word2vec = Word2VecFeatureExtractor()
    features.append(('ngram', vect))
    features.append(('ngram1', vect1))
    #features.append(('word2vec',word2vec))
    features.append(('custom_features', custom_features))
    return features


# SVM Linear classifier gives the best score - better than ensemble

# In[9]:

def all_classifiers():
    #clf1 = linear_model.LogisticRegression(C=3, max_iter=3000, tol=1e-8)
    clf3 = svm.SVC(kernel='linear', gamma=1.2, C=1, decision_function_shape="ovo",probability=True)
    #clf4 = linear_model.SGDClassifier(n_iter=2000,loss = 'modified_huber', penalty = 'elasticnet',alpha=0.001, n_jobs=-1)
    #eclf = VotingClassifier(estimators=[('lr',clf1),('svm_rbf',clf3), ('sgd' , clf4)], voting="soft")
    return clf3


# ### Pipeline

# In[10]:

from sklearn.feature_selection import SelectPercentile
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
best_clf = Pipeline([
    ('all', FeatureUnion(all_features())),
    ('linear',all_classifiers()),
    ])


# ### Testing pipeline and custom features

# In[11]:

df_train, df_test = load_Dataset()


# In[12]:

best_clf.fit(df_train.Comment,df_train.Insult)
predicted = best_clf.predict(df_test.Comment)


# In[13]:

accuracy_score(df_test.Insult,predicted)


# In[14]:

class_labels = np.sort(df_train.Insult.unique())
lables = [str(i) for i in class_labels]
print(classification_report(df_test.Insult, predicted, target_names=lables))


# In[15]:

best_clf.predict_proba(["'we let you in'? Who's we? YOU didn't let anyone in. You are a nobody, you have no say, remember? Nobody that is actually responsible or has a say on who comes and goes shares you're views. That's why 'we' are here. Don't ever forget or let someone tell you otherwise. End of discussion."])


# In[16]:

best_clf.predict_proba(["fuk"])


# #### Testing on validation set - using pipeline and custom

# In[17]:

df_train, df_test = load_Dataset("test1")


# In[18]:

predictions = best_clf.predict(df_test.Comment)


# In[19]:

accuracy_score(df_test.Insult,predictions)


# In[20]:

print(classification_report(df_test.Insult, predictions, target_names=lables))


# #### Testing on test set - using pipeline and custom

# In[21]:

df_train1, df_test1 = load_Dataset("test")


# In[22]:

predictions1 = best_clf.predict(df_test1.Comment)


# In[23]:

accuracy_score(df_test1.Insult,predictions1)


# In[24]:

print(classification_report(df_test1.Insult, predictions1, target_names=lables))


# * 1,6 for both, max_features = all - train 85.29 test 86.36 validation 73.82% --> Vivek's list of abuses
# * 1,6 for both, max_features = all - train 85.55 test 86.59 validation 74.18% --> Nihar's list of abuses

# In[27]:

import pickle

#dup to pickle
with open('model_demo.pkl', 'wb') as f:
    pickle.dump(best_clf, f)
#Load the model from this pickle later
best_clf = pickle.load(open('model_demo.pkl','rb'))

# returns confidence score ()
def is_abuse(model,sentence):
    confidence = model.predict_proba([sentence])
    print(confidence)
    return confidence


sentence = "you fuck!"
# calling the function
is_abuse(best_clf,sentence)


# In[ ]:




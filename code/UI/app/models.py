import tweepy
import pickle
import nltk
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import words as w
from nltk.stem import WordNetLemmatizer


CONSUMER_KEY = "STFpndP02iqX61Q02gSwWSeju"
CONSUMER_SECRET = "bIzoT7jRXxP0jHYK7U2l6BWCZzqz3vOfWrO7WA8EdkOkbY9K7P"
ACCESS_TOKEN = "797174826007834625-hr4WaBsASLneOpHv7vjb2TidFNljkKr"
ACCESS_TOKEN_SECRET = "lTQEGUk5j7gSn7egSoWdSi8i0Fy2PwcjYiWH1MmHIaO0R"

stopwords = ['a', 'an', 'the', 'or', 'is', 'are', 'was', 'were', 'have']

def build_tokens(text):
    '''
    Returns tokenized text given a sentence as input
    '''
    tweetTokenizer = nltk.tokenize.TweetTokenizer(
        strip_handles=True, reduce_len=True)
    tokens = tweetTokenizer.tokenize(text)
    return tokens


class AdditionalFeatureExtractor(BaseEstimator, TransformerMixin):
    '''
    Class for extracting custom features
    '''

    def __init__(self):
        pass

    def get_feature_names(self):
        return (['percent_bad', 'vader_compound', 'num_words', 'vader_neg', 'vader_pos'])

    def num_bad(self, df):
        '''
        Get number of words in each sentence
        '''
        num_words = [len(word) for word in df]
        '''
        Get percent of abusive words in each sentence
        '''
        with open("list_of_abuses.txt", "r") as abuse_list:
            abuses = abuse_list.read().split()
            num_abuses = 0
            for abuse in abuses:
                num_abuses += 1
            # number of badwords in list of abuses
            num_bad = [np.sum([word.lower().count(abuse) for abuse in abuses])
                       for word in df]
            norm_bad = np.array(num_bad) / np.array(num_words, dtype=np.float)
        return norm_bad

    def num_words(self, df):
        '''
        Get number of words in each sentence
        '''
        num_words = [len(word) for word in df]
        return num_words

    def vader_helper(self, df):
        '''
        Vader analysis
        '''
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['compound'])
        return vader_feature

    def vader_helper_neg(self, df):
        '''
        Vader analysis
        '''
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['neg'])
        return vader_feature

    def vader_helper_pos(self, df):
        '''
        Vader analysis
        '''
        sid = SentimentIntensityAnalyzer()
        vader_feature = []
        for sentence in df:
            ss = sid.polarity_scores(sentence)
            vader_feature.append(ss['pos'])
        return vader_feature

    def transform(self, df, y=None):
        '''
        Add all features to an array
        '''
        X = np.array([self.num_bad(df), self.vader_helper(df), self.num_words(
            df), self.vader_helper_neg(df), self.vader_helper_pos(df)]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)

    def fit(self, df, y=None):
        return self


def all_features():
    '''
    Feature extractor function
    '''
    features = []
    custom_features = AdditionalFeatureExtractor()
    vect = TfidfVectorizer(sublinear_tf=True, ngram_range=(
        1, 6), analyzer="char", stop_words=stopwords, tokenizer=build_tokens)
    vect1 = TfidfVectorizer(sublinear_tf=True, ngram_range=(
        1, 6), analyzer="word", stop_words=stopwords, tokenizer=build_tokens)

    features.append(('ngram', vect))
    features.append(('ngram1', vect1))
    features.append(('custom_features', custom_features))
    return features


def is_abuse(sentence):
    '''
    Checks if the sentence passed in the input is abusive
    '''
    model = pickle.load(open('model.pkl', 'rb'))
    confidence = model.predict_proba([sentence])
    return confidence[0][1]


def retrieve_abusive_tweets(twitter_handle):
    '''
    Initializes the tweepy twitter API and retreives recent tweets
    '''
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    recent_tweets = []
    for status in tweepy.Cursor(api.user_timeline, id=twitter_handle).items(1000):
        recent_tweets.append(status.text.encode('utf-8').decode('utf-8'))

    model = pickle.load(open('model.pkl', 'rb'))
    confidence = model.predict_proba(recent_tweets)

    # Get abusive tweets
    abusive_tweets = {}
    for count in range(0,len(recent_tweets)):
        if confidence[count][1] > 0.50:
            abusive_tweets[recent_tweets[count]] = confidence[count][1]
    # Get the 10 most abusive tweets
    sorted_abusive_tweets = sorted(abusive_tweets, key=abusive_tweets.get, reverse=True)

    return sorted_abusive_tweets, recent_tweets

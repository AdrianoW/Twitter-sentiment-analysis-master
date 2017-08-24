# -*- coding: utf-8 -*-

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import numpy as np
import re

# elongated words
elong_regex = re.compile(r"(.)\1{2}")
elong_punct_regex = re.compile(r"(!|\?|\?!|!\?)\1{1}")

# emoticons regex
emoticon_string_pos = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\'^]?                 # optional nose
      [\)\]dDpP\:\}\|\\] # mouth      
      |
      [\(\[\:\{@\|\\]  # mouth
      [\-o\*\'^]?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
emoticon_pos_regex = re.compile(emoticon_string_pos, re.VERBOSE | re.I | re.UNICODE)
emoticon_string_neg = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\'^]?                 # optional nose
      [\(\[/\:\{@\|\\] # mouth      
      |
      [\)\]dDpP/\:\}\|\\] # mouth
      [\-o\*\'^]?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
emoticon_neg_regex = re.compile(emoticon_string_neg, re.VERBOSE | re.I | re.UNICODE)

negated_tokens = r"""
    (?:
        ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
        )$
    )
    |
    n't"""
negated_token_regex = re.compile(negated_tokens, re.VERBOSE | re.I | re.UNICODE)
end_negated = r"^[.!?,;:]+$"
end_negated_regex = re.compile(end_negated, re.VERBOSE | re.I | re.UNICODE)

time_str = r"\dp.|\d/\d|\d:\d"
time_regex = re.compile(time_str, re.VERBOSE | re.I | re.UNICODE)

stop_words = set(stopwords.words('english'))

def tokenize(tweet, pType='informal'):
    '''
    Tokenize the tweet's text
    Params:
        tweet: tweet text
    returns:
        list of tokens
    '''
    if pType == 'informal':
        tknzr = TweetTokenizer()
        return tknzr.tokenize(tweet)
    elif pType == 'normal':
        return nltk.word_tokenize(tweet)
    elif pType == 'space':
        return tweet.split(' ')

def bigrams(tweet):
    '''
    Bigrams in the text
    Params:
        tweet: tokens representing tweet
    '''
    return list(nltk.bigrams(tweet))

def trigrams(tweet):
    '''
    Bigrams in the text
    Params:
        tweet: tokens representing tweet
    '''
    return list(nltk.trigrams(tweet))

def ngrams(tweet, n=4):
    '''
    Bigrams in the text
    Params:
        tweet: tokens representing tweet
    '''
    return list(nltk.ngrams(tweet, n))
    
def non_contiguous(tokens, non_cont_char='---'):
    '''
    Create len(ngrams) with words replaced by *
    Params:
        tokens: the tokenized tweet
    '''
    ret = []
    all_tokens = ngrams(tokens)+ngrams(tokens,3)
    for ngram in all_tokens:
        # substitute each word individually by ---
        for i in range(len(ngram)):
            # create a new vector to hole the info
            tmp = list(ngram)
            tmp[i] = non_cont_char
            
            # save them joined again with the --- next to its neighbors
            jtmp = " ".join(tmp)
            jtmp = jtmp.replace(non_cont_char+" ", non_cont_char).replace(" "+non_cont_char, non_cont_char)
            ret.append(jtmp)

    return ret

def char_ngrams(tweet, n):
    '''
    Return the ngrams of chars
    Params:
        tweet: tokens representing tweet
    '''
    return [tweet[i:i+n] for i in range(len(tweet)-n+1)]

def count_all_caps(tweet):
    '''
    Count the number of all caps
    Params:
        tweet: tokens representing tweet
    '''
    return np.sum([ 1 if w.isupper() else 0 for w in tweet])

def count_mentions(tweet):
    '''
    Count the number of mentions
    Params:
        tweet: tokens representing tweet
    '''
    return np.sum([ 1 if w.find('@')==0 else 0 for w in tweet])

def count_hash(tweet):
    '''
    Count the number of hashtags
    Params:
        tweet: tokens representing tweet
    '''
    return np.sum([ 1 if w.find('#')==0 else 0 for w in tweet])

def count_pos(pos):
    '''
    Count the number of each POS
    Params:
        tweet: tokens representing tweet
    '''
    #pos = nltk.pos_tag(tweet)
    d = {}
    for k,v in pos:
        d['POS_'+v] = d.get(v,0) +1
    return d

def find_elongated(tweet):
    '''
    Return the elongated words
    Params:
        tweet: tokens representing tweet
    '''
    return [word for word in tweet if elong_regex.search(word)]
    
def find_elongated_punct(tweet):
    '''
    Return the elongated words
    Params:
        tweet: tokens representing tweet
    '''
    return [word for word in tweet if elong_punct_regex.search(word)]
    
def normalize_mentions(tweet):
    '''
    Normalize the number of mentions
    Params:
        tweet: tokens representing tweet
    '''
    if type(tweet) == list:
        return [ '@MENTION' if w.find('@')==0 else w for w in tweet]
    else:
        return '@MENTION' if tweet.find('@')==0 else tweet

def normalize_hash(tweet):
    '''
    Normalize hashtags
    Params:
        tweet: tokens representing tweet
    '''
    if type(tweet) == list:
        return [ '#HASH' if w.find('#')==0 else w for w in tweet]
    else:
        return '#HASH' if tweet.find('#')==0 else tweet
    
def normalize_url(tweet):
    '''
    Normalizer urls
    Params:
        tweet: tokens representing tweet
    '''
    if type(tweet) == list:
        return [ 'HTTP' if w.find('http')==0 else w for w in tweet]
    else:
        return 'HTTP' if tweet.find('http')==0 else tweet

def normalize_lower(tweet):
    '''
    Normalizer urls
    Params:
        tweet: tokens representing tweet
    '''
    if type(tweet) == list:
        return [ w.lower() for w in tweet]
    else:
        return tweet.lower()

def remove_stop(tweet, p_stopwords=None):
    '''
    Remove stop word or return list of removed stop words
    '''
    if not p_stopwords:
        sw = stop_words
    else:
        sw = p_stopwords

    if type(tweet) == list:
        return [word for word in tweet if word not in sw]
    else:
        if tweet not in sw:
            return tweet

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def normalize_number(tweet):
    '''
    Normalizer urls
    Params:
        tweet: tokens representing tweet
    '''
    if type(tweet) == list:
        return [ 'NUMBER' if is_number(w) else w for w in tweet]
    else:
        return 'NUMBER' if is_number(w) else tweet
    
def convert_token(token):
    '''
    Check if it is a single token or a tuple or list
    '''
    if type(token) == str or type(token)==unicode:
        key = token
    else:
        key = ' '.join(token)
        
    return key

def negate_tokens(tw):
    """
    Negate the tokens of the tweets
    """
    new_tokens = []
    neg = False
    for token in tw:
        # check if this words starts the negated process
        if negated_token_regex.search(token):
            neg = True
        elif end_negated_regex.search(token):
            neg = False
        # check if this token removes the negation process
        elif neg:
            token = 'NEG_'+ token
            
        new_tokens.append(token)
    return new_tokens
    
class LexiconStats(object):
    
    def __init__(self, lexicon, prefix, thresh=0, ptype=None):
        '''
        Compute the stats for a set of lexicons applied to tokens
        Params:
            lexicon: a class of the lexicon that must implement get_values
            prefix: the prefix for the created columns
            tresh: treshold to be considered positive or negative
        '''
        self._lexicon = lexicon
        self._prefix = prefix
        self._thresh = thresh
        self._type = ptype
        
    def get_stats(self, tokens):
        '''
        Create the stats based on the tokens
        Params:
            tokens: list of tokens.
        Return:
            total count: 
            total score:
            max score:
            score last:
        '''
        # create the base values
        prefix = self._prefix
        typ = self._type
        
        # get the polarity values. The tokens are first joined with ' '
        if typ:
            base = [ self._lexicon.get_polarity(convert_token(t), typ).values()[0] for t in tokens]
        else:
            base = [ self._lexicon.get_polarity(convert_token(t)).values()[0] for t in tokens]
        
        # create the statistics. if there is none, return 0 vector
        array = np.array(base)
        data = {}
        if len(base)!=0:
            data[prefix.upper()+'_MAX'] = np.max(base)
            data[prefix.upper()+'_SUM'] = np.sum(base)
            data[prefix.upper()+'_COUNT'] = np.size(array[array>0])
            data[prefix.upper()+'_LASPOS'] = base[-1]>0
        else:
            data[prefix.upper()+'_MAX'] = 0
            data[prefix.upper()+'_SUM'] = 0
            data[prefix.upper()+'_COUNT'] = 0
            data[prefix.upper()+'_LASPOS'] = 0
            
        return data
    
# From positive and negative to 1 and 0        
encode_label = {'positive':1, 'negative':-1, 'neutral':0, 'unknwn':None, 
                'objective-OR-neutral':0 , 'objective':0}
decode_label = {1:'positive', -1:'negative', 0:'neutral'}

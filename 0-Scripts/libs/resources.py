# -*- coding: utf-8 -*-
#Original script
# Copyright (c) 2013 Tobias Günther and Lenz Furrer.
# All rights reserved.
#
# The 3 classes WordCluster, SentiWordNet and Wordlist were made by Tobias and Lenz and
# adapted to my structure. Please refer to:
#    PDF: http://aclweb.org/anthology/S/S13/S13-2054.pdf
#    BibTex: http://aclweb.org/anthology/S/S13/S13-2054.bib
#

#
# Author: Adriano W Almeida
# date: 2015
#

from files import TupleReader, TabReader
from os.path import join
import pandas as pd
from utils import jar_wrapper, prefix_dict
from . import RESOURCES_DIR

# Lower and Higher margins for positive and negative
LOW_BOUND = -.1
HIGH_BOUND = .1

class WordCluster(object):
    def __init__(self):
        cluster_mapping = {}
        folder = 'CMUTaggerPos'
        # read the file and create a dict with the values for each word
        file_name = join(RESOURCES_DIR,
                         folder,
                         '50mpaths2')
        with open(file_name) as f:
            for line in f:
                if not line.strip():
                    continue
                cluster, word, count = line.strip().split('\t')
                cluster_mapping[word] = cluster
        self.cluster_mapping = cluster_mapping

    def get_cluster(self, word):
        return self.cluster_mapping.get(word, None)


class SentiWordNet(object):
    def __init__(self):
        pos_score = {}
        neg_score = {}

        with open(RESOURCES_DIR + "SentiWordNet_3.0.0_20130122.txt") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                postag, wid, pos, neg, synset, exmpl = line.split('\t')

                for token in synset.split():
                    word = token.split('#')[0]

                    pos_score[word] = float(pos)
                    neg_score[word] = float(neg)

        self.pos_score = pos_score
        self.neg_score = neg_score

    def get_score(self, word):
        return self.pos_score.get(word, 0), self.neg_score.get(word, 0)


class WordList(object):
    """
    Handles the Wordlist of english words
    http://www-01.sil.org/linguistics/wordlists/english/
    """
    def __init__(self):
        wl = set()

        with open(RESOURCES_DIR + "wordsEn.txt") as f:
            # read the files and add to a set
            for line in f:
                line = line.strip()

                if not line:
                    continue

                wl.add(line)

        self.wl = wl

    def __contains__(self, word):
        # check if the word exists in the set
        return word in self.wl


class Synesketch(object):
    """
    Class that will handle Synesketch analysis of a frase
    """
    def __init__(self):
        pass

    def get_score(self, text):
        '''
        For every text, return the value
        '''
        args = [RESOURCES_DIR + 'synesketch2.jar', '"{}"'.format(text)]
        result = jar_wrapper(*args)
        try:
            return int(result[0])
        except:
            return 0


class AFINN_111(object):
    '''
    Manages AFFINN_111 file
    '''

    def __init__(self):
        '''
        Load the file and create a dictionary
        '''
        folder = 'AFINN-111'
        tr = TupleReader(join(RESOURCES_DIR, folder, 'AFINN-111.txt'))
        self._afinn = tr()

    def get_score_val(self, word):
        '''
        Return AFINN-111 score value as int. 0 if it does not exists
        '''
        return int(self._afinn.get(word, 0))

    def get_score_str(self, word):
        '''
        Return AFINN-111 score value as str. None if it does not exists
        '''
        return self._afinn.get(word, None)


class NRCSentiment140(object):
    '''
    Wrapper for the NRC sentiment 140
    https://raw.githubusercontent.com/mikemeding/NLP-Project/master/Corpus/searchTermLexicon/Sentiment140/pairs-pmilexicon.txt
    https://raw.githubusercontent.com/mikemeding/NLP-Project/master/Corpus/searchTermLexicon/Sentiment140/bigram-pmilexicon.txt
    https://raw.githubusercontent.com/mikemeding/NLP-Project/master/Corpus/searchTermLexicon/Sentiment140/unigrams-pmilexicon.txt
    -----------
    Details of the lexicon can be found in the following peer-reviewed
    publication:

    -- In Proceedings of the seventh international workshop on Semantic
    Evaluation Exercises (SemEval-2013), June 2013, Atlanta, Georgia, USA.

    BibTeX entry:
    @InProceedings{MohammadKZ2013,
      author    = {Mohammad, Saif and Kiritchenko, Svetlana and Zhu, Xiaodan},
      title     = {NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets},
      booktitle = {Proceedings of the seventh international workshop on Semantic Evaluation Exercises (SemEval-2013)},
      month     = {June},
      year      = {2013},
      address   = {Atlanta, Georgia, USA}
    }
    .......................................................................
    '''
    def __init__(self):
        folder = 'NRC-Sentiment140'
        # read the unigram, bigram and pairs
        rd = TabReader(join(RESOURCES_DIR, folder,
                            'unigrams-pmilexicon.txt'))
        self._uni = rd(4)
        rd = TabReader(join(RESOURCES_DIR, folder,
                            'bigrams-pmilexicon.txt'))
        self._bi = rd(4)
        rd = TabReader(join(RESOURCES_DIR, folder,
                            'pairs-pmilexicon.txt'))
        self._pairs = rd(4)
        
        # the field names
        self._fields = ['SENT140_SCORE', 'SENT140_NUM_POS', 'SENT140_NUM_NEG']
        self._empty = self._create_dict()
    
    def _create_dict(self, triple=None):
        '''
        Returns a filled structure with the values
        '''
        if not triple:
            triple = [0, 0, 0]
        ret = [ float(v) for v in triple]
        return dict(zip(self._fields, ret))

    def get_score(self, word, ptype='unigram'):
        '''
        Returns (sentimentScore,numPositive,numNegative)
        ---------------------------------------------------------
        sentimentScore is a real number. A positive score indicates
            positive sentiment. A negative score indicates negative sent.
        numPositive is the number of times the term co-occurred with a
            positive marker such as a positive emoticon
        numNegative is the number of times the term co-occurred with a
            negative marker such as a negative emoticon

        Params:
            word: String to search for
            type: unigram, bigram, pairs
        '''
        assert ptype in ['unigram', 'bigram', 'pairs'], 'Type should be ' \
            'unigram, bigram or pairs. Got {}'.format(ptype)

        # check the kind of word it is
        if ptype == 'unigram':
            return self._create_dict(self._uni.get(word, None))
        elif ptype == 'bigram':
            return self._create_dict(self._bi.get(word, None))
        else:
            return self._create_dict(self._pairs.get(word, None))
            
    def get_polarity(self, word, ptype='unigram'):
        '''
        Returns the polarity in a range [-10, 0, 10]
        '''
        ret = self.get_score(word, ptype)
        return {'SENT140_POL':ret['SENT140_SCORE']}

class NRCHashtagSentiment(object):
    '''
    NRC Hashtag Sentiment lexicon.
    http://saifmohammad.com/WebPages/Abstracts/NRC-SentimentAnalysis.htm
    @InProceedings{MohammadKZ2013,
      author    = {Mohammad, Saif M. and Kiritchenko, Svetlana and Zhu, Xiaodan},
      title     = {NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets},
      booktitle = {Proceedings of the seventh international workshop on Semantic Evaluation Exercises (SemEval-2013)},
      month     = {June},
      year      = {2013},
      address   = {Atlanta, Georgia, USA}
      }
    '''
    def __init__(self):
        folder = 'NRC-hashtag'
        # read the unigram, bigram and pairs
        rd = TabReader(join(RESOURCES_DIR, folder, 'unigrams-pmilexicon.txt'))
        self._uni = rd(4)
        rd = TabReader(join(RESOURCES_DIR, folder, 'bigrams-pmilexicon.txt'))
        self._bi = rd(4)
        rd = TabReader(join(RESOURCES_DIR, folder, 'pairs-pmilexicon.txt'))
        self._pairs = rd(4)
        
        self._fields = ['NRCH_SCORE', 'NRCH_NUM_POS', 'NRCH_NUM_NEG']
        self._empty = self._create_dict()
    
    def _create_dict(self, triple=None):
        '''
        Returns a filled structure with the values
        '''
        if not triple:
            triple = [0, 0, 0]
        ret = [ float(v) for v in triple]
        return dict(zip(self._fields, ret))

    def get_score(self, word, ptype='unigram'):
        '''
        Returns (sentimentScore,numPositive,numNegative)
        ---------------------------------------------------------
        sentimentScore is a real number. A positive score indicates
            positive sentiment. A negative score indicates negative sent.
        numPositive is the number of times the term co-occurred with a
            positive marker such as a positive emoticon
        numNegative is the number of times the term co-occurred with a
            negative marker such as a negative emoticon

        Params:
            word: String to search for
            ptype: unigram, bigram, pairs
        '''
        assert ptype in ['unigram', 'bigram', 'pairs'], 'Type should be ' \
            'unigram, bigram or pairs. Got {}'.format(ptype)

        # check the kind of word it is
        if ptype == 'unigram':
            return self._create_dict(self._uni.get(word, None))
        elif ptype == 'bigram':
            return self._create_dict(self._bi.get(word, None))
        else:
            return self._create_dict(self._pairs.get(word, None))
            
    def get_polarity(self, word, ptype='unigram'):
        '''
        Returns the polarity in a range [-10, 0, 10]
        '''
        ret = self.get_score(word, ptype)
        return {'NRCH_POL':ret['NRCH_SCORE']}
    

class NRCEmotion(object):
    '''
    NRC Manually constructed emotion lexicon
    http://www.saifmohammad.com/WebDocs/README-NRC-Lex.txt
    
    Made of 6 different emotional fields plus 2 for negative and positive emotions
    '''
    def __init__(self):
        folder = 'NRC-Emotion'
        # read the file and create a dict with the values for each word
        file_name = join(RESOURCES_DIR,
                         folder,
                         'lexicon-wordlevel-alphabetized-v0.92.txt')
        df = pd.read_csv(file_name, sep='\t', skiprows=46,
                         names=['word', 'sentiment', 'value'])
        df['sentiment'] = df['sentiment'].apply(lambda v: 'NRCE_'+v.upper())
        self._emotion = df.pivot(index='sentiment', columns='word',
                                 values='value').to_dict()

        # an empty dictionary
        self._empty = {'NRCE_ANGER': 0,
                       'NRCE_ANTICIPATION': 0,
                       'NRCE_DISGUST': 0,
                       'NRCE_FEAR': 0,
                       'NRCE_JOY': 0,
                       'NRCE_NEGATIVE': 0,
                       'NRCE_POSITIVE': 0,
                       'NRCE_SADNESS': 0,
                       'NRCE_SURPRISE': 0,
                       'NRCE_TRUST': 0}
                       
        # to easily return polarity
        self._polarity_map = {'11': 0,
                              '10':1,
                              '01':-1,
                              '00':0}

    def get_score(self, word, type='unigram'):
        '''
        Returns dict : {'NRCE_ANGER': 0,
                       'NRCE_ANTICIPATION': 0,
                       'NRCE_DISGUST': 0,
                       'NRCE_FEAR': 0,
                       'NRCE_JOY': 0,
                       'NRCE_NEGATIVE': 0,
                       'NRCE_POSITIVE': 0,
                       'NRCE_SADNESS': 0,
                       'NRCE_SURPRISE': 0,
                       'NRCE_TRUST': 0}
        Params:
            word: String to search for
        '''

        # return the values
        return self._emotion.get(word, self._empty)
        
    def get_polarity(self, word):
        '''
        Returns the polarity in a range [-1,0,1]
        '''
        ret = self.get_score(word)
        ret = self._polarity_map[str(ret['NRCE_POSITIVE'])+str(ret['NRCE_NEGATIVE'])]
        return {'NRCE_POL':ret}


class BingOpinion(object):
    '''
    Bing's opinion lexicon
    http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    The lexicon is composed 1 positive list and another negative with many words, even mispelled ones.
    '''
    def __init__(self):
        folder = 'Bing'
        # read the file and create a dict with the values for each word
        file_name = join(RESOURCES_DIR,
                         folder,
                         'negative-words.txt')
        self._neg_score = {}
        with open(file_name) as f:

            # ignore initial lines
            for _ in range(35):
                f.next()

            for word in f:
                self._neg_score[word.replace('\n', '')] = 1

        # read the positive words
        self._pos_score = {}
        file_name = join(RESOURCES_DIR,
                         folder,
                         'positive-words.txt')
        with open(file_name) as f:

            # ignore initial lines
            for _ in range(35):
                f.next()

            for word in f:
                self._pos_score[word.replace('\n', '')] = 1
                
        # to easily return polarity
        self._polarity_map = {'11': 0,
                              '10':1,
                              '01':-1,
                              '00':0}

    def get_score(self, word):
        '''
        Returns a dict with the positive and negative flags
        Params:
            word: the word to be searched
        Returns:
            {BING_POS: value, BING_NEG:value}
        '''
        ret = {'BING_POS': self._pos_score.get(word, 0),
               'BING_NEG': self._neg_score.get(word, 0) }
        return ret
        
    def get_polarity(self, word):
        '''
        Returns the polarity in a range [-1,0,1]
        '''
        ret = self.get_score(word)
        ret = self._polarity_map[str(ret['BING_POS'])+str(ret['BING_NEG'])]
        return {'BING_POL':ret}
        

class MPQA(object):
    '''
    MPQA - subjectivity clues
    Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual
    Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005,
    Vancouver, Canada.

    Returns:
        a. type - either strongsubj or weaksubj
            A clue that is subjective in most context is considered strongly
            subjective (strongsubj), and those that may only have certain
            subjective usages are considered weakly subjective (weaksubj).

        b. len - length of the clue in words
            All clues in this file are single words.

        c. word1 - token or stem of the clue

        d. pos1 - part of speech of the clue, may be anypos (any part of speech)

        e. stemmed1 - y (yes) or n (no)
            Is the clue word1 stemmed?  If stemmed1=y, this means that the
            clue should match all unstemmed variants of the word with the
            corresponding part of speech.  For example, "abuse", above, will
            match "abuses" (verb), "abused" (verb), "abusing" (verb), but not
            "abuse" (noun) or "abuses" (noun).

        f. priorpolarity - positive, negative, both, neutral
            The prior polarity of the clue.  Out of context, does the
            clue seem to evoke something positive or something negative.
    '''
    def __init__(self):
        folder = 'MPQA'
        # read the file and create a dict with the values for each word
        file_name = join(RESOURCES_DIR,
                         folder,
                         'subjclueslen1-HLTEMNLP05.tff')

        with open(file_name) as f:
            self._mpqa = {}

            # create a dict with this line
            for line in f:
                ex = {}
                line = line.strip()
                fields = line.split(' ')
                for f in fields:
                    # there is a bug in the file, word1=pervasive, deal with it
                    try:
                        k, v = f.split('=')
                        ex[k] = int(v) if v.isdigit() else v
                    except:
                        pass

                # check if this is the last line
                if ex.get('word1', None):
                    # save it
                    self._mpqa[ex['word1']] = ex

        # create and empty structure
        self._empty = {'len': 0,
                       'pos1': '',
                       'priorpolarity': '',
                       'stemmed1': '',
                       'type': '',
                       'word1': ''}
                       
        self._polarity_map = {'positive':1, 'negative':-1}

    def get_score(self, word):
        '''
        Returns the raw values inside the file
        '''
        return self._mpqa.get(word, self._empty)
        
    def get_polarity(self, word):
        '''
        Return dict with the polarity, with a renamed field
        '''
        ret = self._mpqa.get(word, self._empty)['priorpolarity']
        return {'MPQA_POL': self._polarity_map.get(ret, 0)}
        
class GeneralInquirer(object):
    
    def __init__(self):
        folder = 'GeneralInquirer'
        # read the file
        file_name = join(RESOURCES_DIR,
                         folder,
                         'inquireraugmented.csv')
                         
        # create a dataframe with it
        self._inq = pd.read_csv(file_name, sep=";").set_index("Entry").T.to_dict()
        
    def get_score(self, word):
        # returns the values for the word
        return self._inq.get(word, None)

########################################################################################
# new part
########################################################################################
import json
from config import RESOURCES_DIR
from os.path import join
import numpy as np

import parse as p

def process_lex(lex, tokens_list, bigrams=False, trigrams=False, non_contiguous=False, join_char=' ', cont_char='---',
                return_tokens=False, return_avg=False, return_sum=False, return_min=False, return_max=False,
                return_pol_amt=False, final_pol=False
               ):
    """
    Process a list of lists of tokens using a given lexicon
    Params:
        lex: the lexicon structure created by me
        tokens_list: a [[t1, t2, ...], [t1, t3,...], ...]
        bigrams: should use bigrams
        trigrams: should use trigrams
        join_char: the character that should be used when joining the bigrams and trigrams
    """
    lex_tokens = []
    
    # statistics of the token
    # create all
    for tl in tokens_list:
        tokens = {}
        
        # statistics vars
        summ=0.
        avg=0.
        minn=0.
        maxn=0.
        pneg=0
        ppos=0
        
        # copy list
        new_tl = list(tl)
        # check if bigrams and trigrams should be created
        if bigrams:
            new_tl += [ join_char.join(t)  for t in p.bigrams(tl) ]
        if trigrams:
            new_tl += [ join_char.join(t)  for t in p.trigrams(tl) ]
        if non_contiguous:
            # create non contiguous of 4 tokens
            new_tl += [ t for t in p.non_contiguous(tl, cont_char) ]

        # for each token, get the polarity
        ret_tokens = process_tokens(new_tl, lex)

        # check if the tokens and polarities should be returned as features
        if return_tokens:
            # add tokens to the features
            tokens.update(ret_tokens)
            
        if ret_tokens:
             # make a value vector for the statistics
            vals = np.array([ v for k, v in ret_tokens.iteritems()])
            avg = vals.mean()
            summ = vals.sum()
            minn = vals.min()
            maxn = vals.max()
            pneg = sum(vals<0)
            ppos = sum(vals>0)
            

        if return_avg:
            tokens[lex.prefix+'_avg'] = avg
        if return_sum:
            tokens[lex.prefix+'_sum'] = summ       
        if return_min:
            tokens[lex.prefix+'_min'] = minn       
        if return_max:
            tokens[lex.prefix+'_max'] = maxn      
        if return_pol_amt:
            tokens[lex.prefix+'_negamt'] = pneg  
            tokens[lex.prefix+'_posamt'] = ppos  
        if final_pol:
            fpol = 0
            if summ>0:
                fpol = 1
            elif summ<0:
                fpol = -1
            
            tokens[lex.prefix+'_fpol'] = fpol

        lex_tokens.append(tokens)
        
    return lex_tokens

def convert_emotional_token(token, thresh=None):
    ret = {}

    # sum the sentiment values of all words
    for w, e in token.iteritems():
        for k,v in e.iteritems():
            if (not thresh) or (abs(v)>thresh):
                ret[w+'_'+k] = v
    return ret

def process_emo_lex(lex, tokens_list, bigrams=False, trigrams=False, non_contiguous=False, join_char=' ', cont_char='---',
                return_tokens=False, return_avg=False, return_sum=False, return_min=False, return_max=False,
                return_pol_amt=False, filter_tags=[]
               ):
    """
    Process a list of lists of tokens using a given lexicon
    Params:
        lex: the lexicon structure created by me
        tokens_list: a [[t1, t2, ...], [t1, t3,...], ...]
        bigrams: should use bigrams
        trigrams: should use trigrams
        join_char: the character that should be used when joining the bigrams and trigrams
    """
    lex_tokens = []
    
    # statistics of the token
    # create all
    for tl in tokens_list:
        tokens = {}
        
        # statistics vars
        summ=0.
        avg=0.
        minn=0.
        maxn=0.
        pamnt=0
        
        # copy list
        new_tl = list(tl)
        # check if bigrams and trigrams should be created
        if bigrams:
            new_tl += [ join_char.join(t)  for t in p.bigrams(tl) ]
        if trigrams:
            new_tl += [ join_char.join(t)  for t in p.trigrams(tl) ]
        if non_contiguous:
            # create non contiguous of 4 tokens
            new_tl += [ t for t in p.non_contiguous(tl, cont_char) ]

        # for each token, get the polarity dictionary containing all the emotions
        ret_tokens = process_tokens(new_tl, lex)
        
        # check if the tokens and polarities should be returned as features
        if return_tokens:
            # add tokens to the features
            tokens.update(convert_emotional_token(ret_tokens))
            
        if ret_tokens:
            # make a value vector for the statistics
            vals = {}
            for token, emo_dict in ret_tokens.iteritems():
                # add the token to the list of words with tokens
                curr_tokens = vals.get('token', [])
                curr_tokens.append(token)
                vals['token'] = curr_tokens
                
                # for each of the sentiments, add to its inidividual list
                for e,v in emo_dict.iteritems():
                    if filter_tags and e not in filter_tags:
                        continue
                    curr_emo_vals = vals.get(e, [])
                    curr_emo_vals.append(v)
                    vals[e] = curr_emo_vals
                    
            # for each of the emotion lists, do the sats
            stats = {}
            for k,vls in vals.iteritems():
                if k=='token':
                    continue
                emo_vals = np.array(vls)
                stats[k+'_avg'] = emo_vals.mean()
                stats[k+'_summ'] = emo_vals.sum()
                stats[k+'_minn'] = emo_vals.min()
                stats[k+'_maxn'] = emo_vals.max()
#                 stats[k+'_pneg'] = sum(emo_vals<0)
                stats[k+'_count'] = len(emo_vals)
                stats[k+'_vals'] = emo_vals
                #print vls
                #print stats
            
            #for k,v in stats.iteritems():
            #    if 'vals' in k:
            #        tokens[k] = v
            
            if return_avg:
                for k,v in stats.iteritems():
                    if 'avg' in k:
                        tokens[k] = v
            if return_sum:
                for k,v in stats.iteritems():
                    if 'summ' in k:
                        tokens[k] = v
            if return_min:
                for k,v in stats.iteritems():
                    if 'minn' in k:
                        tokens[k] = v
            if return_max:
                for k,v in stats.iteritems():
                    if 'max' in k: 
                        tokens[k] = v
            if return_pol_amt:
                for k,v in stats.iteritems():
                    if ('count' in k) :#or ('ppos' in k):
                        tokens[k] = v

        lex_tokens.append(tokens)
        
    return lex_tokens

class BaseLexicon(object):
    """
    Base class that holds the minimal information needed to properly
    deal with a tweet and save stats
    """
    def __init__(self, prefix, bigrams=False, trigrams=False, non_contiguous=False, 
                 radicals=False, opinion=True, grams_separator=' ', join_char=' ',
                 cont_char='---'):
        # save what is contained in this lexicon
        self.prefix = prefix
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.non_contiguous = non_contiguous
        self.radicals = radicals
        self.opinion = opinion
        self.join_char = join_char
        self.cont_char = cont_char
        self.negated = False
        
        # the lexicon info
        self.data = None
        self.df = None
        self.token_num = 0
        
        # stats info
        self.reset_stats()
        self.processed = False
        self.grams_separator = grams_separator
        
        # define the best params to process this lex
        self.best_features = {
            'return_tokens':True,
            'return_avg':True,
            'return_sum':True,
            'return_min':True,
            'return_max':True,
            'return_pol_amt':True
        }
        self.selection_percent = None
        
    def _load_lexicon_json(self, file_name):
        """
        loads the preprocessed json
        """
        # construct path
        file_name = join(RESOURCES_DIR,
                         'processed',
                         file_name)
        
        # load the json
        with open(file_name, 'r')  as f:
            self.data = json.load(f)
        
        # fill other structures
        self.token_num = len(self.data.keys())
        self._create_df()
        
    def _create_df(self):
        """
        Creates a DF from the data. Prefix all names
        """
        self.df = pd.DataFrame.from_dict(self.data, orient='index')
        self.df.columns = [ self.prefix+"_"+c if c!=0 else self.prefix for c in self.df.columns ]
            
    def compare_lexicons(self, other):
        """
        Compares this lexicon COVERAGE with another one
        """
        # for now, just a simple difference
        data = other.data
        same = set(data.keys()).intersection(self.data.keys())
        diff = set(data.keys()).difference(self.data.keys())
        per_same = (len(same)*100.0)/self.token_num
        
        # check the r² between metrics
        if False:
            print "{}\t{}\t{}".format(self.prefix, other.prefix, "Common")
            print "{}\t{}\t{}\n".format(self.token_num, 
                                        len(data.keys()), 
                                        len(same))
        return {self.prefix:{other.prefix: per_same}}
    
    def correlate_lexicon(self, other):
        """
        Calculates the CORRELATION between 2 OPINION lexicons
        """
        if self.opinion and other.opinion:
            temp_df = self.df.join(other.df)
            temp_df.dropna(inplace=True)
            corr = temp_df.corr()

            return {corr.columns[0]: {corr.columns[1]: corr.iloc[0,1]}}
        else:
            return None
    
    def process_tokens(self, tokens_list):
        """
        Process the tweet over single token
        Params:
            Tweet Class
        Returns:
            a list of tokens found
        """
        tokens = []
        for token in tokens_list:
            # get the value for this token
            ret = self.data.get(token, None)
            if ret:
                tokens.append({token: ret})

            # TODO: need to deal with negation
        
        return tokens

    def check_lex_sent_match(self, val, sent):
        """
        Check if the value and the sentiment match
        Params:
            val: sum of all the tokens' associated polarity
            sent: tweet sentiment - positive, negative ...
        """
        if (val > HIGH_BOUND and sent == "positive") or \
           (val < LOW_BOUND and sent == "negative") or \
           (val > LOW_BOUND and val < HIGH_BOUND and sent == "neutral"):
            return 1
        else:
            return 0

    def process_lex(self, tokenized_list, bigrams=None, trigrams=None, non_contiguous=None, 
                          join_char=None, cont_char=None, use_best_features=False, **kwargs):
        """
        Given a list of tokenized tweets, process them creating a list of dictionaries
        It uses the params set on lexicon initialization.

        """
        # check if should use the previous calculated best params
        if use_best_features:
            params = self.best_features
            params['bigrams'] = params.get('bigrams', self.bigrams)
            params['trigrams'] = params.get('trigrams', self.trigrams)
            params['non_contiguous'] = params.get('non_contiguous', self.non_contiguous)
            params['join_char'] = params.get('join_char', self.join_char)
            params['cont_char'] = params.get('cont_char', self.cont_char)
        else:
            params = kwargs
            params['bigrams'] = bigrams if bigrams != None else self.bigrams
            params['trigrams'] = trigrams if trigrams != None else self.trigrams
            params['non_contiguous'] = non_contiguous if non_contiguous != None else self.non_contiguous
            params['join_char'] = join_char if join_char != None else self.join_char
            params['cont_char'] = cont_char if cont_char != None else self.cont_char

        # check what kind of lex is this and process accordingly
        if self.opinion:
            return process_lex(self, tokenized_list,
                               **params)
        else:
            return process_emo_lex(self, tokenized_list,
                               **params)


    def process_tweet(self, tweet):
        """
        Process a tweet
        Params:
            Tweet Class
        Returns:
            returns the total sum for the lexicon and the value of each token
        """
        self.processed = True

        # process the basic tokens
        sent_list = []
        tokens = self.process_tokens(tweet.tokens)
        self.total_words_match += len(tokens)
        self.total_words += len(tweet.tokens)
        sent_list += tokens

        # save the tokens that we found
        tweet.save_tokens_found(self.prefix, tokens)

        tokens = self.process_tokens(tweet.bigrams)
        self.total_bigrams_match += len(tokens)
        self.total_bigrams += len(tweet.bigrams)
        sent_list += tokens

        # save the tokens that we found
        tweet.save_bigrams_found(self.prefix, tokens)

        tokens = self.process_tokens(tweet.trigrams)
        self.total_trigrams_match += len(tokens)
        self.total_trigrams += len(tweet.trigrams)
        sent_list += tokens

        # save the tokens that we found
        tweet.save_trigrams_found(self.prefix, tokens)

        tokens = self.process_tokens(tweet.non_contiguous)
        self.total_contiguous_match += len(tokens)
        self.total_contiguous += len(tweet.non_contiguous)
        sent_list += tokens

        # save the tokens that we found
        tweet.save_non_contiguous_found(self.prefix, tokens)
        
        # # process bigrams
        # if self.bigrams:
        #     for token in tweet.bigrams:
        #         # get the value for this token
        #         ret = self.data.get(token, None)
        #         if ret:
        #             self.total_bigrams_match += 1
        #             sent_list.append({token: ret})
        # self.total_bigrams += len(tweet.bigrams)

        # # process trigrams
        # if self.trigrams:
        #     for token in tweet.trigrams:
        #         # get the value for this token
        #         ret = self.data.get(token, None)
        #         if ret:
        #             self.total_trigrams_match += 1
        #             sent_list.append({token: ret})
        # self.total_trigrams += len(tweet.trigrams)

        # # process non_contiguous
        # if self.non_contiguous:
        #     for token in tweet.non_contiguous:
        #         # get the value for this token
        #         ret = self.data.get(token, None)
        #         if ret:
        #             self.total_contiguous_match += 1
        #             sent_list.append({token: ret})
        # self.total_contiguous += len(tweet.non_contiguous)

        
        self.total_tweets +=1
        
        # check if this tweet had any match
        if len(sent_list)>0:
            ret = None
            match = None
            self.total_tweets_match += 1 
            
            # check if this an opinion lexicon
            if self.opinion:
                summ = 0
                for sent in sent_list:
                    summ += sent.values()[0]

                ret = {'SUM_'+self.prefix: summ}, sent_list
                match = self.check_lex_sent_match(summ, tweet.sent)
            else:
                # it is a sentiment lexicon. Sum the sentiments individually
                sum_dic = {}
                for ws in sent_list:
                    # sum the sentiment values of all words
                    for w, e in ws.iteritems():
                        for k,v in e.iteritems():
                            sum_dic[k] = sum_dic.get(k, 0) + v

                # check the match for the sentiment
                match = {}
                for k,v in sum_dic.iteritems():
                    match[k] = self.check_lex_sent_match(v, tweet.sent)
                
                # prefix the dictionary keys before returning it
                ret = prefix_dict(sum_dic, self.prefix+'_'), sent_list

            # save the final features created for this lexicon and if it matches the sentiment
            tweet.save_lex_total_feaures(self.prefix, ret[0])
            tweet.save_lex_sent_match(self.prefix, match)
            return ret
        else:
            # no term found in the dictionary
            return None, None
    
    def print_match(self):
        """
        Print the variables that are related with the match
        """
        # print the variables
        print "Total tweets match %d" % self.total_tweets_match
        print "Total words match %d" % self.total_words_match
        print "Total bigrams match %d" % self.total_bigrams_match
        print "Total trigrams match %d" % self.total_trigrams_match
        print "Total contiguous match %d" % self.total_contiguous_match

    def get_match_stats(self):
        """
        Creates a dictionary with the stats of use of this lexicon
        """
        # print self.total_words, self.total_tweets
        ret = {
            'words': self.total_words_match,
            'tweets': self.total_tweets_match,
            '%_words': float(self.total_words_match)/self.total_words,
            '%_tweets': float(self.total_tweets_match)/self.total_tweets
        }

        if self.bigrams:
            ret['bigrams'] = self.total_bigrams_match
            ret['%_bigrams'] = float(self.total_bigrams_match)/self.total_bigrams
        if self.trigrams:
            ret['trigrams'] = self.total_trigrams_match
            ret['%_trigrams'] = float(self.total_trigrams_match)/self.total_trigrams
        if self.non_contiguous:
            ret['non_contiguous'] = self.total_contiguous_match
            ret['%_non_contiguous'] = float(self.total_contiguous_match)/self.total_contiguous
        return {self.prefix: ret}

    def get_total_stats(self):
        """
        Returns general statistics like number of tweets
        Ususally it will be called just by one lexicon as it is the same for all
        """
        return { 
            'TOTAL': {
                'words': self.total_words,
                'tweets': self.total_tweets,
                'bigrams': self.total_bigrams,
                'trigrams': self.total_trigrams,
                'contiguous': self.total_contiguous,
                'words': self.total_words
                }
            }

    def reset_stats(self):
        """
        Reset all the statistics for this lexicon
        """
        # stats info
        self.total_words = 0
        self.total_tweets = 0
        
        # general information on processed tokens
        self.total_bigrams = 0
        self.total_trigrams = 0
        self.total_contiguous = 0
        self.total_words = 0

        # match info
        self.total_tweets_match = 0
        self.total_words_match = 0
        self.total_bigrams_match = 0
        self.total_trigrams_match = 0
        self.total_contiguous_match = 0

class AnewLexicon(BaseLexicon):
    """
    Holds the information for AnewLexicon
    """
    def __init__(self):
        super(AnewLexicon, self).__init__(prefix='ANEW', opinion=True)
        self._load_lexicon_json('anew.json')

        self.best_features = {
            'return_tokens':True,
            'return_avg':True,
            'return_sum':True,
            'return_min':True,
            'return_max':True,
            'return_pol_amt':True,
            'final_pol':True
        }
        self.selection_percent = 30
        self.negated = False
        
anew = AnewLexicon()

class BingLexicon(BaseLexicon):
    """
    Holds Bing and Liu Lexicon Info
    """
    def __init__(self):
        super(BingLexicon, self).__init__(prefix='BING')
        self._load_lexicon_json('bing.json')

        # define the best params to use this lexicon
        self.best_features = {'return_tokens':False,
                            'return_avg':True,
                            'return_sum':True,
                            'return_min':True,
                            'return_max':True,
                            'return_pol_amt':True,
                            'final_pol':True}

        self.selection_percent = None
        self.negated = True
        
bing = BingLexicon()
        
class DALLexicon(BaseLexicon):
    """
    Holds the information for AnewLexicon
    """
    def __init__(self):
        super(DALLexicon, self).__init__(prefix='DAL', opinion=False)
        self._load_lexicon_json('dal.json')

        # define the best params to use this lexicon
        self.best_features = dict(return_tokens=False, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True,
              bigrams=True, trigrams=True, filter_tags=['pleasantness'])
        
dal = DALLexicon()

class MPQALexicon(BaseLexicon):
    """
    Holds Bing and Liu Lexicon Info
    """
    def __init__(self):
        super(MPQALexicon, self).__init__(prefix='MPQA')
        self._load_lexicon_json('mpqa.json')

        # define the best params to use this lexicon
        self.best_features = {  'return_tokens':False,
                                'return_avg':True,
                                'return_sum':True,
                                'return_min':True,
                                'return_max':True,
                                'return_pol_amt':True,
                                'final_pol':True}
        self.selection_percent = None

mpqa = MPQALexicon()
        
class MSOLLexicon(BaseLexicon):
    """
    Holds Bing and Liu Lexicon Info
    """
    def __init__(self):
        super(MSOLLexicon, self).__init__(prefix='MSOL', bigrams=True, trigrams=True, join_char='_',
                                          non_contiguous=True, cont_char='-')
        self._load_lexicon_json('msol.json')

        # define the best parms
        self.best_features = {
            'return_tokens':True,
            'return_avg':True,
            'return_sum':True,
            'return_min':True,
            'return_max':True,
            'return_pol_amt':True,
            'final_pol':True
            } 
        self.selection_percent = 5
        
msol = MSOLLexicon()

class NRCHashLexicon(BaseLexicon):
    """
    Holds Bing and Liu Lexicon Info
    """
    def __init__(self):
        super(NRCHashLexicon, self).__init__(prefix='NRCHASH', bigrams=True, non_contiguous=True,
                                             trigrams=True)
        self._load_lexicon_json('nrc_hash.json')

        # define the best parms
        self.best_features = {
            'return_tokens':False,
            'return_avg':True,
            'return_sum':True,
            'return_min':True,
            'return_max':True,
            'return_pol_amt':True,
            'final_pol':True,
            'bigrams': False,
            'trigrams': False,
            'non_contiguous': True
            } 
        self.selection_percent = None
        self.negated = False

nhash = NRCHashLexicon()

class Sent140Lexicon(BaseLexicon):
    """
    Holds Bing and Liu Lexicon Info
    """
    def __init__(self):
        super(Sent140Lexicon, self).__init__(prefix='SENT140', bigrams=True, trigrams=True, 
                                             non_contiguous=True)
        self._load_lexicon_json('sent140.json')

        # define the best parms
        self.best_features = {
            'return_tokens':False,
            'return_avg':True,
            'return_sum':True,
            'return_min':True,
            'return_max':True,
            'return_pol_amt':True,
            'final_pol':True
            } 
        self.selection_percent = None
        self.negated = False

s140 = Sent140Lexicon()

class SentiStrenghtLexicon(BaseLexicon):
    """
    Holds Bing and Liu Lexicon Info
    """
    def __init__(self):
        super(SentiStrenghtLexicon, self).__init__(prefix='SSTREN')
        self._load_lexicon_json('sentstrenght.json')

        # define the best parms
        self.best_features = {
            'return_tokens':False,
            'return_avg':False,
            'return_sum':False,
            'return_min':False,
            'return_max':True,
            'return_pol_amt':False,
            'final_pol':False 
        }
        self.selection_percent = None

ss = SentiStrenghtLexicon()

class TSLexStrengthLexicon(BaseLexicon):
    """
    Holds TS Lex Info
    """
    def __init__(self):
        super(TSLexStrengthLexicon, self).__init__(prefix='TSLEX', bigrams=True, \
            trigrams=True, join_char=' ')
        self._load_lexicon_json('ts_lex.json')

        # define the best parms
        self.best_features = {
            'return_tokens':False,
            'return_avg':True,
            'return_sum':True,
            'return_min':True,
            'return_max':True,
            'return_pol_amt':True,
            'final_pol':True
        }
        # self.selection_percent = 5

tslex = TSLexStrengthLexicon()

class WNALexicon(BaseLexicon):
    """
    Holds WNA Info
    """
    def __init__(self):
        super(WNALexicon, self).__init__(prefix='WNA', opinion=False,
                                         bigrams=True, trigrams=True, join_char='_')
        self._load_lexicon_json('wna.json')

        # define the best parms
        self.best_features = dict(return_tokens=True, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True)

        self.selection_percent = 5

wna = WNALexicon()

class SenticNetLexicon(BaseLexicon):
    """
    Holds Sentic Net Info
    """
    def __init__(self):
        super(SenticNetLexicon, self).__init__(prefix='SENTN', bigrams=True, opinion=False,
                                               trigrams=True, join_char=' ')
        self._load_lexicon_json('senticnet.json')

        # define the best parms
        self.best_features = dict(return_tokens=True, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True,
              bigrams=True, trigrams=True, join_char=' ')
        self.selection_percent = 15
        

sentn = SenticNetLexicon()

class EmoLexicon(BaseLexicon):
    """
    Holds Sentic Net Info
    """
    def __init__(self):
        super(EmoLexicon, self).__init__(prefix='EMOLX', opinion=False)
        self._load_lexicon_json('emolex.json')

        # define the best parms
        self.best_features = dict(return_tokens=False, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True)


emlx = EmoLexicon()

class SentiSenseLexicon(BaseLexicon):
    """
    Holds Sentic Net Info
    """
    def __init__(self):
        super(SentiSenseLexicon, self).__init__(prefix='SENTS', opinion=False, bigrams=False,
                grams_separator='_')
        self._load_lexicon_json('sentisense.json')

        # define the best parms
        self.best_features = dict(return_tokens=False, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True, filter_tags=['sadness'])
        
    def process_lex(self, tokens_list, bigrams=None, trigrams=None, non_contiguous=None, 
                          join_char=None, cont_char=None, use_best_features=None,
                          return_tokens=None, return_avg=None, return_sum=None, return_min=None, 
                          return_max=None, return_pol_amt=None, filter_tags=[]):

        """
        Process a list of lists of tokens using a given lexicon
        Params:
            lex: the lexicon structure created by me
            tokens_list: a [[t1, t2, ...], [t1, t3,...], ...]
            bigrams: should use bigrams
            trigrams: should use trigrams
            join_char: the character that should be used when joining the bigrams and trigrams
        """
        # full the default config in case nothing was passed
        bigrams = self.bigrams 
        trigrams = self.trigrams 
        non_contiguous = self.non_contiguous if not non_contiguous else non_contiguous

        # check if best features should be used
        if use_best_features:
            return_tokens = self.best_features['return_tokens'] 
            return_avg = self.best_features['return_avg'] 
            return_sum = self.best_features['return_sum'] 
            return_min = self.best_features['return_min'] 
            return_max = self.best_features['return_max']
            return_pol_amt = self.best_features['return_pol_amt'] 
            filter_tags = self.best_features['filter_tags'] 

        # create all
        lex_tokens = []
        for tl in tokens_list:
            tokens = {}
            
            # statistics vars
            summ=0.
            avg=0.
            minn=0.
            maxn=0.
            pamnt=0
            
            # copy list
            new_tl = list(tl)
            
            # check if bigrams and trigrams should be created
            if bigrams:
                new_tl += [ self.grams_separator.join(t)  for t in p.bigrams(tl) ]
            
            # get the POS
            pos = nltk.pos_tag(new_tl)
            
            # add the pos to find the values on SentiSense   
            
            new_tl = [ u'{}.{}'.format(tag_pos[0], u.penn_to_wn(tag_pos[1])) for tag_pos in pos]
            ret_tokens = process_tokens(new_tl, self)
            #except:
                #print(tl)
                #print(pos)
                #print([ tag_pos[0]+'.'+u.penn_to_wn(tag_pos[1]) for tag_pos in pos])
            #    ret_tokens = {}
            
            
            # remove the pos tag
            #new_tokens = {}
            #for token in ret_tokens.iteritems():
            #    tmp = token[0].split('#')[0]
            #    new_tokens[tmp] = token[1]
            #ret_tokens = new_tokens
            
            # check if the tokens and polarities should be returned as features
            if return_tokens:
                # add tokens to the features
                tokens.update(convert_emotional_token(ret_tokens))
                
            if ret_tokens:
                # make a value vector for the statistics
                vals = {}
                for token, emo_dict in ret_tokens.iteritems():
                    # add the token to the list of words with tokens
                    curr_tokens = vals.get('token', [])
                    curr_tokens.append(token)
                    vals['token'] = curr_tokens
                    
                    # for each of the sentiments, add to its inidividual list
                    for e,v in emo_dict.iteritems():
                        if filter_tags and e not in filter_tags:
                            continue
                        curr_emo_vals = vals.get(e, [])
                        curr_emo_vals.append(v)
                        vals[e] = curr_emo_vals
            
                # for each of the emotion lists, do the sats
                stats = {}
                for k,vls in vals.iteritems():
                    if k=='token':
                        continue
 
                    emo_vals = np.array(vls)
                    stats[k+'_avg'] = emo_vals.mean()
                    stats[k+'_summ'] = emo_vals.sum()
                    stats[k+'_minn'] = emo_vals.min()
                    stats[k+'_maxn'] = emo_vals.max()
    #                 stats[k+'_pneg'] = sum(emo_vals<0)
                    stats[k+'_count'] = sum(emo_vals>0)

                if return_avg:
                    for k,v in stats.iteritems():
                        if 'avg' in k:
                            tokens[k] = v
                if return_sum:
                    for k,v in stats.iteritems():
                        if 'summ' in k:
                            tokens[k] = v
                if return_min:
                    for k,v in stats.iteritems():
                        if 'minn' in k:
                            tokens[k] = v
                if return_max:
                    for k,v in stats.iteritems():
                        if 'max' in k: 
                            tokens[k] = v
                if return_pol_amt:
                    for k,v in stats.iteritems():
                        if ('count' in k) :#or ('ppos' in k):
                            tokens[k] = v

            lex_tokens.append(tokens)
        
        return lex_tokens

ssense = SentiSenseLexicon()

class LewLexicon(BaseLexicon):
    """
    Holds LEW Info
    """
    def __init__(self):
        super(LewLexicon, self).__init__(prefix='LEW', opinion=False, bigrams=True,
                grams_separator='-')
        self._load_lexicon_json('lew.json')

        # define the best parms
        self.best_features = dict(return_tokens=False, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True, filter_tags=['Evaluation'])



    def process_lex(self, tokens_list, bigrams=None, trigrams=None, non_contiguous=None, 
                          join_char=None, cont_char=None, use_best_features=None,
                          return_tokens=None, return_avg=None, return_sum=None, return_min=None, 
                          return_max=None, return_pol_amt=None, filter_tags=[]):

        """
        Process a list of lists of tokens using a given lexicon
        Params:
            lex: the lexicon structure created by me
            tokens_list: a [[t1, t2, ...], [t1, t3,...], ...]
            bigrams: should use bigrams
            trigrams: should use trigrams
            join_char: the character that should be used when joining the bigrams and trigrams
        """
        # full the default config in case nothing was passed
        bigrams = self.bigrams 
        trigrams = self.trigrams 
        non_contiguous = self.non_contiguous if not non_contiguous else non_contiguous

        # check if best features should be used
        if use_best_features:
            return_tokens = self.best_features['return_tokens'] 
            return_avg = self.best_features['return_avg'] 
            return_sum = self.best_features['return_sum'] 
            return_min = self.best_features['return_min'] 
            return_max = self.best_features['return_max']
            return_pol_amt = self.best_features['return_pol_amt']
            filter_tags = self.best_features['filter_tags']

        # create all
        lex_tokens = []
        for tl in tokens_list:
            tokens = {}
            
            # statistics vars
            summ=0.
            avg=0.
            minn=0.
            maxn=0.
            pamnt=0
            
            # copy list
            new_tl = list(tl)
            
            # check if bigrams and trigrams should be created
            if bigrams:
                new_tl += [ self.grams_separator.join(t)  for t in p.bigrams(tl) ]
            
            # get the POS
            pos = nltk.pos_tag(new_tl)
            
            # add the pos to find the values on lew
            new_tl = [ tag_pos[0]+'#'+u.penn_to_lew(tag_pos[1]) for tag_pos in pos]
            ret_tokens = process_tokens(new_tl, self)
            
            # remove the pos tag
            new_tokens = {}
            for token in ret_tokens.iteritems():
                tmp = token[0].split('#')[0]
                new_tokens[tmp] = token[1]
            ret_tokens = new_tokens
            
            # check if the tokens and polarities should be returned as features
            if return_tokens:
                # add tokens to the features
                tokens.update(convert_emotional_token(ret_tokens))
                
            if ret_tokens:
                # make a value vector for the statistics
                vals = {}
                for token, emo_dict in ret_tokens.iteritems():
                    # add the token to the list of words with tokens
                    curr_tokens = vals.get('token', [])
                    curr_tokens.append(token)
                    vals['token'] = curr_tokens
                    
                    # for each of the sentiments, add to its inidividual list
                    for e,v in emo_dict.iteritems():
                        if filter_tags and e not in filter_tags:
                            continue
                        curr_emo_vals = vals.get(e, [])
                        curr_emo_vals.append(v)
                        vals[e] = curr_emo_vals
                        
                # for each of the emotion lists, do the sats
                stats = {}
                for k,vls in vals.iteritems():
                    if k=='token':
                        continue
                    emo_vals = np.array(vls)
                    stats[k+'_avg'] = emo_vals.mean()
                    stats[k+'_summ'] = emo_vals.sum()
                    stats[k+'_minn'] = emo_vals.min()
                    stats[k+'_maxn'] = emo_vals.max()
    #                 stats[k+'_pneg'] = sum(emo_vals<0)
                    stats[k+'_count'] = sum(emo_vals>0)

                if return_avg:
                    for k,v in stats.iteritems():
                        if 'avg' in k:
                            tokens[k] = v
                if return_sum:
                    for k,v in stats.iteritems():
                        if 'summ' in k:
                            tokens[k] = v
                if return_min:
                    for k,v in stats.iteritems():
                        if 'minn' in k:
                            tokens[k] = v
                if return_max:
                    for k,v in stats.iteritems():
                        if 'max' in k: 
                            tokens[k] = v
                if return_pol_amt:
                    for k,v in stats.iteritems():
                        if ('count' in k) :#or ('ppos' in k):
                            tokens[k] = v

            lex_tokens.append(tokens)
            
        return lex_tokens


lew = LewLexicon()

class LewEmoLexicon(LewLexicon):
    
    def __init__(self):
        super(LewEmoLexicon, self).__init__()
        
        self.prefix ='LEWEmo'
        self._load_lexicon_json('lew_emo.json')
        

        # define the best parms
        self.best_features = dict(return_tokens=False, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True, filter_tags=['Evaluation'])

lewEmo = LewEmoLexicon()    

class EmoSenticNetLexicon(BaseLexicon):
    """
    Holds LEW Info
    """
    def __init__(self):
        super(EmoSenticNetLexicon, self).__init__(prefix='EMOSNET', opinion=False, bigrams=True,
                grams_separator='_')
        self._load_lexicon_json('emosnet.json')

        self.best_features = dict(return_tokens=False, return_avg=True, 
              return_sum=True, return_min=True, return_max=True,
              return_pol_amt=True, filter_tags=['Disgust','Joy'])

emosnet = EmoSenticNetLexicon()

from nltk.corpus import sentiwordnet
import nltk
from collections import defaultdict
import libs.utils as u

class SentiWordNetLexicon(BaseLexicon):
    """
    Holds SWN Info
    """
    def __init__(self):
        super(SentiWordNetLexicon, self).__init__(prefix='SWN')

        # define the best parms
        self.best_features = {
            'return_tokens':False,
            'return_avg':True,
            'return_sum':True,
            'return_thresh':True,
            'return_decision':True,
            'use_median_word':False,
            'thresh':.75
            }
        self.selection_percent = None
        self.negated = True

    def get_synset_values(self, token_pos):
        """
        Returns a formatted dictionary with the pos, neg and vals of the synset
        """
        # construct the synset 
        ret = None
        pos = u.penn_to_wn(token_pos[1])
        
        # check if there is the NEG flag
        if 'NEG_' in token_pos[0]:
            token = token_pos[0].replace('NEG_', '')
        else:
            token = token_pos[0]
        
        # check if it is an unknown position
        if not pos:
            return None
        synset = '.'.join((token_pos[0], pos, '01'))
        
        # get values. In case not found, return None
        try:
            vals = sentiwordnet.senti_synset(synset)
            ret = [vals.pos_score(), vals.neg_score(), vals.obj_score()]
        except:
            None
        return ret 

    def get_median_synset_values(self, token_pos):
        """
        Returns a formatted dictionary with the pos, neg and vals of the synset
        """
        # construct the synset 
        ret = None
        pos = u.penn_to_wn(token_pos[1])
        
        # check if there is the NEG flag
        if 'NEG_' in token_pos[0]:
            token = token_pos[0].replace('NEG_', '')
        else:
            token = token_pos[0]
            
        # check if it is an unknown position
        if not pos:
            return None
        
        # get values. In case not found, return empty dict
        try:
            synsets = sentiwordnet.senti_synsets(token, pos)
            ret = np.array([(s.pos_score(), s.neg_score(), s.obj_score())  for s in synsets]).mean(axis=0).tolist()
        except:
            None
        return ret

    def process_lex(self, tokens_list, return_tokens=False, return_sum=True, 
                    return_avg=True, return_thresh=True, thresh=.75,
                    return_decision=True, use_median_word=False, use_best_features=False):
        lex_tokens = []
        
        # check if should use the previous calculated best params
        # check if best features should be used
        if use_best_features:
            return_tokens = self.best_features['return_tokens'] 
            return_sum = self.best_features['return_sum'] 
            return_avg = self.best_features['return_avg'] 
            return_thresh = self.best_features['return_thresh']
            thresh = self.best_features['thresh']
            return_decision = self.best_features['return_decision']
            use_median_word = self.best_features['use_median_word']
        
        if use_median_word:
            func = self.get_median_synset_values
        else:
            func = self.get_synset_values

        # create all
        for tl in tokens_list:
            features = {}
            tokens_dict = defaultdict(float)
            count = 0
            pos_score = 0
            neg_score = 0
            obj_score = 0
            pos_score_tre = 0
            neg_score_tre = 0  
            count_tre =0 
            
            # create a list of the pos for the tokens
            pos = nltk.pos_tag(tl)
            
            # for each token find the pos, neg and obj value
            for p in pos:
                synset = func(p)
                    
                if type(synset) == list:
                    # in case it is a NEGated, invert positive and negative
                    if 'NEG_' in p[0]:
                        synset[0]= synset[1]
                        synset[1]= synset[0]
                    
                    # add the info to the stats
                    count += 1
                    pos_score += synset[0]
                    neg_score += synset[1]
                    obj_score += synset[2]
                    
                    # a threshold will make only not objective words to be considered
                    if synset[2] < thresh:
                        pos_score_tre += synset[0]
                        neg_score_tre += synset[1]
                        count_tre +=1
                    
                    # save the values for this token
                    tokens_dict[self.prefix+'_POS_'+p[0]] = synset[0]
                    tokens_dict[self.prefix+'_NEG_'+p[0]] = synset[1]
                    tokens_dict[self.prefix+'_OBJ_'+p[0]] = synset[2]
                
            if return_tokens:
                features.update(tokens_dict)

            if return_sum:
                # calculate the final stats
                features[self.prefix+'_pos_tot'] = pos_score
                features[self.prefix+'_neg_tot'] = neg_score
                features[self.prefix+'_obj_tot'] = obj_score
            if return_avg:
                if count==0:
                    count = 1
                features[self.prefix+'_pos_avg'] = pos_score/count
                features[self.prefix+'_neg_avg'] = neg_score/count
                
            if return_thresh:
                # calculate the final stats
                if count_tre==0:
                    count_tre = 1
                features[self.prefix+'_pos_avg_tre'] = pos_score_tre/count_tre
                features[self.prefix+'_neg_avg_tre'] = neg_score_tre/count_tre
                
            if return_decision:
                if count==0:
                    count = 1 
                pos_avg = pos_score/count
                neg_avg = neg_score/count
                if pos_avg > neg_avg:
                    features[self.prefix+'_DECISION'] = 1
                if pos_avg < neg_avg:
                    features[self.prefix+'_DECISION'] = -1
                else:
                    features[self.prefix+'_DECISION'] = 0
                
            # save features    
            lex_tokens.append(dict(features))
        
        return lex_tokens
swn = SentiWordNetLexicon()



# create an array of all lexicons for easier bulk manipulation
# ss
lexs = [bing, swn, msol, ss, nhash, s140, tslex, mpqa,\
        anew, wna, dal, sentn, emlx, ssense, lew, emosnet]

def get_lexs_stats():
    """
    Returns the stats for all the lexicons
    """
    ret = {}

    # traverse the lexicons getting the stats
    for lex in lexs:
        if lex.processed:
            ret.update(lex.get_match_stats())

    # get the total stats from one of the lexicons
    ret.update(lexs[0].get_total_stats())
    return ret

def get_lexs_stats_df():
    """
    Returns the stats for all the lexicons
    """
    return pd.DataFrame(get_lexs_stats()).T

def reset_lexs_stats():
    """
    Returns the stats for all the lexicons
    """
    # traverse the lexicons reseting the stats
    for lex in lexs:
        lex.reset_stats()

def process_tokens(tokens_list, lex):
        """
        Process the tweet over single token
        Params:
            Tweet Class
        Returns:
            a list of tokens found
        """
        tokens = {}
        for token in tokens_list:
            # check if there is negation
            neg = False
            if 'NEG_' in token:
                # save the flag that it is negated and remove the negation flag
                neg = True
                new_token = token.replace('NEG_', '')
            else:
                new_token = token

            # get the value for this token
            ret = lex.data.get(new_token, None)
            if ret:
                if type(ret)!=dict:
                    tokens[lex.prefix+'_'+token] = ret if not neg else -ret
                else:
                    tokens[lex.prefix+'_'+token] = ret
        
        return tokens
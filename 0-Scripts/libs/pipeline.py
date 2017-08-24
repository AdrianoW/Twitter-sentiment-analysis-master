# coding=utf-8
from libs.tweet import Tweet
import libs.files as fh
import libs.parse as p
import libs.utils as u
import libs.resources as r
import twokenize as ark
import numpy as np
from sklearn.metrics import f1_score, classification, classification_report, precision_recall_fscore_support
from sklearn.metrics import make_scorer
import sys
from joblib import Parallel, delayed
from sklearn.base import clone
import nltk
import cPickle
from os import path
from sklearn.feature_extraction import DictVectorizer
import scipy as scy
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer

rnd_seed = 9000
K_FOLD = 10

PROCESS_DIR = '../2-Processed'

def create_features(file_name, file_type, cfg):
    """
    Create the features according to config files. 
    """
    features_tweets = []
    sentiment_tweets = []
    tweets = []
    tr = fh.TweetReader(file_name, file_type)

    for t in tr:
        # load the tweet, converting labels to -1,0 and 1
        tw = Tweet(t["text"], t["sentiment"], t["sid"], t["uid"])
        sentiment_tweets.append(p.encode_label.get(t["sentiment"], None))
        tweets.append(t["text"])

        # return the tweets inside a structure
        features_tweets.append(tw)
        
    return features_tweets, sentiment_tweets, tweets

def dump_data(data, name, folder=PROCESS_DIR):
    """
    Dump a file for later use to the processed dir 
    """
    cPickle.dump(data, open(path.join(PROCESS_DIR, name), 'w'))


def load_dump_data(name, folder=PROCESS_DIR):
    """
    Dump a file for later use to the processed dir 
    """
    return cPickle.load(open(path.join(PROCESS_DIR, name), 'r'))


def process_tokens(tweet):
    """
    Create the tokens and remove the stop words
    
    """
    stop_words = set(['the', 'to', 'in', 'on', 'and', 'of', 'a', 'for', 'at', 'with', 'be', 
              'it', 'that', '-', 'this'])
    tknzr = nltk.TweetTokenizer(strip_handles=True)
    tokens = tknzr.tokenize(tweet)
    
    return [token for token in tokens if token not in stop_words]

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
        ret_tokens = r.process_tokens(new_tl, lex)

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

def create_lex_vec(train_lex_features, dev_lex_features, test_lex_features=None):
    """
    Convert the list of dictionaries to a sparse csr matrix
    Params:
        - train_lex_features: list of dictionaries of 
        features created by the train tokens passed over the lex
        - test_lex_features: list of dictionaries of 
        features created by the test tokens passed over the lex
    """
    vec = DictVectorizer()
    tmp_train = vec.fit_transform(train_lex_features)
    tmp_dev = vec.transform(dev_lex_features)
    tmp_test = None
    if test_lex_features:
        tmp_test = vec.transform(test_lex_features)
    
    return tmp_train, tmp_dev, tmp_test, vec

def create_count_vec(train_data, test_data, tokenize=True, tokenizer=ark.tokenizeRawTweetText, verbose=True, test_name='dev',
                     **kwargs):
    """
    Create the train and test data from the data. If tokenizer is null, it will assume that 
    the data is tokenized already, else it will create the tokens using tokenizer over the raw data
    """
    if tokenize:
        count_vect = CountVectorizer(tokenizer=tokenizer, **kwargs)    
    else:
        # count the words but don't tokenize as they are already tokens
        count_vect = CountVectorizer(input='content', lowercase=False,
              tokenizer=lambda x: x)

    # count the words
    train_dataset = count_vect.fit_transform(train_data)
    test_dataset = count_vect.transform(test_data)
    
    if verbose:
        print 'train shape: {}'.format(train_dataset.shape)
        print '{} shape: {}'.format(test_name, test_dataset.shape)
    return train_dataset, test_dataset, count_vect

def clean_tweet(tw, stop_words):
    """
    Clean list of tokens removing words like mentions, numbers, etc
    """
    # remove stop words, normalize mentions, urls and numbers
    tokens = p.remove_stop(tw, p_stopwords=stop_words) 
    tokens = p.normalize_mentions(tokens)
    tokens = p.normalize_url(tokens)
    tokens = p.normalize_number(tokens) 
    #tokens = p.normalize_hash(tokens)
    
    return tokens


import libs.twokenize as ark
stop_words = ['the', 'to', 'in', 'on', 'and', 'of', 'a', 'for', 'at', 'with', 'be', 
              'it', 'that', '-', 'this','-', 'an']
def tokenize_clean_raw(tw, tokenizer=ark.tokenizeRawTweetText, stop_words=stop_words):
    """
    Gets a clean tweet, tokenize and cleans
    """
    tokens = tokenizer(tw)
    tokens = p.normalize_lower(tokens)
    return clean_tweet(tokens, stop_words=stop_words)


def tokenize_negate_clean_raw(tw, tokenizer=ark.tokenizeRawTweetText, stop_words=stop_words):
    """
    Gets a clean tweet, tokenize and cleans
    """
    tokens = tokenizer(tw)
    tokens = p.normalize_lower(tokens)
    tokens = p.negate_tokens(tokens)
    return clean_tweet(tokens, stop_words=stop_words)


from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif, chi2
def auto_select_features(score_func, percentile, train_data, labels, test_data, gold, verbose=True):
    """
    Apply an auto select strategy to the database and return the top 4 algorithms results
    """
    selector = SelectPercentile(score_func, percentile)
    train_sel = selector.fit_transform(train_data, labels)
    test_sel = selector.transform(test_data)
    if verbose:
        print 'Final shape', train_sel.shape, test_sel.shape
    
    return train_sel, test_sel, selector

def get_most_important_feat(selected, vectorizer, top=30):
    """
    Given a selector with the top features and a vectorizes that created a vector, print the top features
    """
    idx = (-selected.scores_).argsort()[:top]
    return np.array(vectorizer.get_feature_names())[idx]

def benchmark_auto_selection(score_func, percentile, train_data, labels, test_data, gold, use_best_params):
    """
    Auto select features and apply the ML algorithm, returning the top 4
    """
    train_sel, test_sel = auto_select_features(score_func, percentile, train_data, labels, test_data, gold)
    ret_sel = run_multiple_class(train_sel, labels, test_sel, gold, use_best_params=use_best_params)
    top = pd.DataFrame(ret_sel[['dev score', 'test score']].nlargest(3, 'dev score'))
    return top.add_prefix(str(percentile)+' ')

def compare_percentiles(score_func, train_data, labels, test_data, gold, use_best_params, percentiles=[5,15,30]):
    """
    Given a set of percentiles, calculate the top 4 and concatenate in a comparative table
    """
    def highlight_max(s):
        '''
        highlight the maximum in a Series in bold.
        '''
        is_max = s == s.max()
        css = ['font-weight: bold' if v else '' for v in is_max]
        return css
    
    # get a first df
    df = benchmark_auto_selection(score_func, percentiles[0], train_data, labels, test_data, gold, use_best_params)
    for per in percentiles[1:]:
        tmp = benchmark_auto_selection(score_func, per, train_data, labels, test_data, gold, use_best_params)
        df = pd.concat([df, tmp], axis = 1)

    return df.style.apply(highlight_max)

def join_lex_features(train_data, train_lex_data, test_data, test_lex_data,
                     verbose=True, create_vec=True):
    """
    Get a matrix of features and join with another 
    Params:
        train_data: the features already created for train
        train_lex_data: list of dictionaries, lexicon features train data
        test_data: the features already created for train
    """
    if create_vec:
        tmp_train, tmp_test, _, _ = create_lex_vec(train_lex_data, test_lex_data)
    else:
        tmp_train = train_lex_data
        tmp_test = test_lex_data

    final_train = scy.sparse.csr_matrix(scy.sparse.hstack((train_data, tmp_train)))
    final_test = scy.sparse.csr_matrix(scy.sparse.hstack((test_data, tmp_test)))
    if verbose:
        print 'train data, lex and final shape: ', train_data.shape, tmp_train.shape, final_train.shape
        print 'test data, lex and final shape: ', test_data.shape, tmp_test.shape, final_test.shape
    return final_train, final_test

def dump_lex_features(lex, train, dev, create_vec=True, test=None):
    """
    Convert the list of dictionaries to a sparse matrix and dumps it
    """
    if create_vec:
        tmp_train, tmp_dev, tmp_test, _ = create_lex_vec(train, dev, test)
    else:
        tmp_train, tmp_dev, tmp_test = train, dev, test
    dump_data(tmp_train, lex.prefix+'_train.pck')
    dump_data(tmp_dev, lex.prefix+'_dev.pck')
    if type(tmp_test) == scy.sparse.csr.csr_matrix:
        dump_data(tmp_test, lex.prefix+'_test.pck')

def load_lex_features_dump(lex):
    """
    Load the saved trained and test lexicon features
    """
    tmp_train = load_dump_data(lex.prefix+'_train.pck')
    tmp_test = load_dump_data(lex.prefix+'_test.pck')
    
    return tmp_train, tmp_test

# how will the models be compared.
def score_func(gold, pred):
    score = f1_score(gold, pred, labels=[-1, 1], average='macro')
    return score
scorer = make_scorer(score_func)

### helpers
def print_thresh_feat(names, vec, thresh):
    for k,v in zip(names, vec.todense().tolist()[0]):
        if v>thresh:
            print k, v


def train_test_model(cls, train_data, labels, test_data, gold, rnd_seed=9000, verbose=0, 
                     test_name = 'dev', k_fold=K_FOLD):
    """
    Given an algorithm (cls), for each fold, it will train and test the model
    
    params:
        cv_dataset: a dataset with the cross_validation structure
        cls: the classifier algorithm
    
    returns:
        a list of the id/dev result/test_result
    """
    if verbose==1:
        print("Training {}".format(cls.__class__.__name__.split('.')[-1]))
    res = []
    fold = model_selection.StratifiedKFold(k_fold, random_state=rnd_seed)
    try:
        for i, (train_idx, dev_idx) in enumerate(fold.split(train_data, labels)):
            # execute the tests
            ret = _fit_and_score(cls, train_data, labels, test_data, gold, train_idx, dev_idx, i,
                verbose=verbose, test_name=test_name)
            res.append(ret)

    except:
        u.print_exception()
    return res

def _fit_and_score(cls, train_data, labels, 
                        test_data, gold, 
                        train_idx, dev_idx, run_id, 
                        verbose=0, test_name='dev'):
    # create the datasets
    tmp_train = train_data if type(train_data) != list else np.array(train_data)
    tmp_test = test_data if type(test_data) != list else np.array(test_data)
    train = tmp_train[train_idx]
    y_train = labels[train_idx]
    dev = tmp_train[dev_idx]
    y_dev = labels[dev_idx]

    # train the model
    model = cls
    model.fit(train, y_train)
    pred_dev = model.predict(dev)

    # predict on the real test dataset
    pred_test = model.predict(tmp_test)

    # measure performance
    dev_score = score_func(y_dev, pred_dev)
    test_score = score_func(gold, pred_test)

    # get the individual performance for each class
    class_score = f1_score(gold, pred_test, labels=[-1, 0, 1], average=None)

    # print the results
    if verbose>1:
        print 'Model {} results: Train - {} / Dev - {}'.format(run_id, 
                                                              dev_score,
                                                              test_score)
    ret = [run_id, dev_score, test_score]
    ret.extend(class_score)
    return ret

                  
def train_test_multi_proc(cls, train_data, labels, test_data, gold, n_jobs=1, verbose=0,
                          pre_dispatch='2*n_jobs', rnd_seed=rnd_seed, k_fold=K_FOLD, test_name='dev'):
    """
    Given an algorithm (cls), for each fold, it will train and test the model
    
    params:
        cv_dataset: a dataset with the cross_validation structure
        cls: the classifier algorithm
    
    returns:
        a list of the id/dev result/test_result
    """
    scores = []
    if verbose==1:
        print("Training {}".format(cls.__class__.__name__.split('.')[-1]) )
    
    
    try:        
        # make the training pipeline    
        fold = model_selection.StratifiedKFold(k_fold, random_state=rnd_seed)

        # create the parallel pipeline
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                            pre_dispatch=pre_dispatch)

        # run the parallel
        scores = parallel(delayed(_fit_and_score)(clone(cls), train_data, labels, test_data, gold,
                                                  train_idx, dev_idx, i, verbose, test_name)
                          for i, (train_idx, dev_idx) in enumerate(fold.split(train_data, labels)))

    except:
        u.print_exception()
        
    if verbose:
        print('Done')
    return scores


def _fit_and_score_single(cls, train_data, labels, 
                               test_data, gold,
                               run_id, rnd_seed ):
    """

    """
    # train the model and predict
    model = cls
    model.set_params(random_state=rnd_seed)
    model.fit(train_data, labels)

    pred_test = model.predict(test_data)
    pred_train = model.predict(train_data)

    # measure performance
    train_score = score_func(labels, pred_train)
    test_score = score_func(gold, pred_test)

    # get the individual performance for each class
    class_score = f1_score(gold, pred_test, labels=[-1, 0, 1], average=None)

    ret = [run_id, train_score, test_score]
    ret.extend(class_score)
    return ret

def predict_test_multi_proc(model_name, train_data, labels, test_data, gold, n_jobs=1, verbose=0,
                          pre_dispatch='2*n_jobs', rnd_seed=rnd_seed, k_fold=K_FOLD, test_name='dev'):
    """
    Given an algorithm (cls), for each fold, it will train and test the model
    
    params:
        cv_dataset: a dataset with the cross_validation structure
        cls: the classifier algorithm
    
    returns:
        a list of the id/dev result/test_result
    """
    scores = []
    if verbose==1:
        print("Training {}".format(cls.__class__.__name__.split('.')[-1]) )
    
    # check what is the model
    if model_name=='LogisticRegression':
        cls = LogisticRegression(**lr_params)
    elif model_name=='SGDClassifier':
        cls = SGDClassifier(**SGD_params)
    else:
        print model_name
        cls = LogisticRegression(**lr_params)

    fold = ( (clone(cls),i) for i in range(0, K_FOLD))

    # do the 10 fold training
    try:        
        # create the parallel pipeline
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                            pre_dispatch=pre_dispatch)

        # run the parallel
        scores = parallel(delayed(_fit_and_score_single)(model, train_data, labels, test_data, gold, i, i)
                          for model, i in fold)
    except:
        u.print_exception()
        
    if verbose:
        print('Done')

    # create the return DataFrame
    model_name = cls.__class__.__name__.split('.')[-1]
    if model_name=='Pipeline':
        model_name = cls.steps[-1][0]
    df = create_scores_df(scores, model_name, test_name)
    return df


from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn.model_selection as model_selection
import pandas as pd
import warnings
from sklearn import preprocessing as pp
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline, make_union

# from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

# lr_params = {'C': 0.3,
#              'class_weight': 'balanced',
#              'dual': False,
#              'fit_intercept': False,
#              'intercept_scaling': 1,
#              'max_iter': 100,
#              'multi_class': 'ovr',
#              'penalty': 'l2',
#              'random_state': 9000,
#              'solver': 'liblinear',
#              'tol': 0.0001,
#              'verbose': 0,
#              'warm_start': False,
#              'random_state':rnd_seed}

lr_params = {'C': 100,
 'class_weight': 'balanced',
 'dual': False,
 'fit_intercept': True,
 'intercept_scaling': 1,
 'max_iter': 300,
 'multi_class': 'ovr',
 'penalty': 'l2',
 'random_state': rnd_seed,
 'solver': 'sag'}

lsvc_params = {'C': 0.5,
'class_weight': 'balanced',
'dual': False,
'fit_intercept': True,
'intercept_scaling': 1,
'loss': 'squared_hinge',
'max_iter': 1000,
'multi_class': 'ovr',
'penalty': 'l2',
'random_state': rnd_seed}

nb_params = {'alpha': 0.01, 'fit_prior': True}

SGD_params = {'alpha': 0.0001,
 'average': False,
 'class_weight': 'balanced',
 'epsilon': 0.1,
 'fit_intercept': True,
 'loss': 'hinge',
 'n_iter': 100,
 'random_state': rnd_seed,
 'shuffle': True}


def create_scores_df(scores, model_name, test_name='dev'):
    """
    Given a model testing return, create a data frame for easier results maninpulation
    """
    col_names = ['runID', 'train score', 
                 '{} score'.format(test_name),\
                 'f1_{}_neg'.format(test_name),\
                 'f1_{}_neu'.format(test_name),\
                 'f1_{}_pos'.format(test_name)]
    data = pd.DataFrame(pd.DataFrame(scores, columns=col_names))
    df = pd.DataFrame(data[col_names[1:]].mean())
    std = pd.DataFrame(data[col_names[1:]].std())
    std.index = ['train std', '{} std'.format(test_name), \
                'f1_{}_neg std'.format(test_name),\
                'f1_{}_neu std'.format(test_name),\
                'f1_{}_pos std'.format(test_name)]
    df = df.append(std)
    df.columns = [model_name]
    return df.transpose()

def create_dummy_scores_df():
    """
    Create a dummy dataframe with 0 values
    """
    return create_scores_df(np.zeros((6,6)), 'Dummy - Error')

def make_col(name, data, f1=False):
    """
    Join a score columns and a standard deviation creating one column only
    """
    if not f1:
        col = "{} score".format(name)
    else:
        col = "{}".format(name)
    return data[col].map(lambda x: '{0:.3f}'.format(x)) +' +- '+ \
           data["{} std".format(name)].map(lambda x: '{0:.3f}'.format(x))

def pprint_results(df, test_name='dev'):
    """
    Pretty print the results with standard deviation
    """
    def highlight_max(s):
        '''
        highlight the maximum in a Series in bold.
        '''
        is_max = orig_data[s.name] == orig_data[s.name].max()
        css = ['font-weight: bold' if v else '' for v in is_max]
        return css
    
    def highlight_top3(s):
        '''
        highlight the top3 in a Series in bold.
        '''
        # colors for the background an create an ordered vector
        shades = [200, 220, 247]
        ordered = orig_data[s.name].sort_values(ascending=False)
        ordered.reset_index(inplace=True, drop=True)
        idx = pd.Index(ordered)
        
        # paint the Background accordingly
        css = ['background-color: rgb({color},{color},{color})'\
               .format(color=shades[idx.get_loc(v)]) \
               if idx.get_loc(v)<3 else '' for v in orig_data[s.name]]
        return css
    
    # create new prittier columns
    orig_data = df
    df = df.copy()
    cols = ['train score', test_name + ' score', \
            'f1_{}_neg'.format(test_name), \
            'f1_{}_neu'.format(test_name), \
            'f1_{}_pos'.format(test_name)]
    
    df['train score'] = make_col('train', df)
    df[test_name+' score'] = make_col(test_name, df)
    
    # create the test_name according to the param name.
    col = 'f1_{}_neg'.format(test_name)
    df[col] = make_col(col, df, True)
    col = 'f1_{}_neu'.format(test_name)
    df[col] = make_col(col, df, True)
    col = 'f1_{}_pos'.format(test_name)
    df[col] = make_col(col, df, True)
    
    # make the return dataframe and apply the styler
    ret = df[cols]
    return ret.style.apply(highlight_max).apply(highlight_top3)

class SparseToArrayTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()

class ArrayToSpareTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return scy.sparse.csr_matrix(X)

class DebugPipe(TransformerMixin):
    def __init__(self):
        self.y = None

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X):
        print 'Transform shape', X.shape
        print 'Y shape', self.y.shape
        return X

class OpinionLexiconTransf(TransformerMixin):
    def __init__(self, lex, params, func=process_lex):
        self.lex = lex
        self.params = params
        self.process_lex = func

    def fit(self, X, y=None):
        sent_features = self.process_lex(self.lex, X, **self.params)
        self.vec = DictVectorizer()
        self.vec.fit(sent_features)
        return self

    def transform(self, X):
        sent_features = self.process_lex(self.lex, X, **self.params)
        ret = self.vec.transform(sent_features)
        return ret

class TokenizeTransform(TransformerMixin):
    def __init__(self, tokenize):
        self.tokenize = tokenize

    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        return [self.tokenize(t) for t in X]



def run_multiple_class(train_data, labels, test_data, gold, n_jobs=6, verbose=0,
                          pre_dispatch='2*n_jobs', rnd_seed=rnd_seed, scale=False, use_best_params=True,
                          classifiers=None, pre_process=[], test_name='dev'):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # check if a list of classifiers was passed. If not, use this.
        if not classifiers:
            if use_best_params:
                mc_classifiers = [
                #    KNeighborsClassifier(3),
                    # SVC(kernel="linear", C=0.025, random_state=rnd_seed),
                #    SVC(random_state=rnd_seed),
                #     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True, random_state=rnd_seed),
                #    DecisionTreeClassifier(random_state=rnd_seed),
                    # RandomForestClassifier(random_state=rnd_seed),
                #     MLPClassifier(alpha=1, random_state=rnd_seed),
                    # AdaBoostClassifier(random_state=rnd_seed),
                #     GaussianNB(),
                #     QuadraticDiscriminantAnalysis(),
                    LogisticRegression(**lr_params), 
                    SGDClassifier(**SGD_params), 
                    # RidgeClassifier(random_state=rnd_seed),
                    #MultinomialNB(**nb_params),
                    LinearSVC(**lsvc_params)]

                sc_classifiers =[]
            else:
                # multi core processor
                mc_classifiers = [
                    KNeighborsClassifier(3),
                    DecisionTreeClassifier(random_state=rnd_seed),
                    RandomForestClassifier(random_state=rnd_seed),
                    AdaBoostClassifier(random_state=rnd_seed), 
                    GaussianNB(),
                    LogisticRegression(random_state=rnd_seed), 
                    SGDClassifier(random_state=rnd_seed), 
                    RidgeClassifier(random_state=rnd_seed),
                    MultinomialNB(),
                    LinearSVC(random_state=rnd_seed)]

                sc_classifiers =[
                    # Pipeline([('toarray', SparseToArrayTransformer()), ('Gaussian', 
                    #     GaussianProcessClassifier(random_state=rnd_seed))]),
                    #MLPClassifier(random_state=rnd_seed),
                    # Pipeline([('toarray', SparseToArrayTransformer()), ('Quadratic', 
                    # QuadraticDiscriminantAnalysis())])
                ]
        else:
            sc_classifiers = classifiers
            mc_classifiers = []
        ret = None

        if scale:
            # scale data to min and max to make sure no feature get more importance
            # http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
            scaler = pp.MaxAbsScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

        all_classifiers = mc_classifiers
        all_classifiers.extend(sc_classifiers)
        for cls in all_classifiers:
            pipe_elem = []
            pipe_elem.extend(pre_process)

            # check if this is the Multinomial 
            if type(cls) == MultinomialNB:
                pipe_elem.append(SparseToArrayTransformer())
                pipe_elem.append(pp.MinMaxScaler())
                pipe_elem.append(ArrayToSpareTransformer())

                # scaler = pp.MinMaxScaler().fit(train_data.toarray())
                # train_data = scaler.transform(train_data.toarray())
                # test_data = scaler.transform(test_data.toarray())
                # train_data = scy.sparse.csr_matrix(train_data)
                # test_data = scy.sparse.csr_matrix(test_data)
            elif type(cls) == SGDClassifier:
                # SGD works better when normalized
                pipe_elem.append(pp.MaxAbsScaler())
                # scaler = pp.MaxAbsScaler().fit(train_data)
                # train_data = scaler.transform(train_data)
                # test_data = scaler.transform(test_data)
            elif type(cls) == GaussianNB:
                pipe_elem.append(SparseToArrayTransformer())

            pipe_elem.append(cls)
            pipeline = make_pipeline(*pipe_elem)

            # run the models and return the tests only, creating a summary dictionary
            if cls in mc_classifiers:
                scores = train_test_multi_proc(pipeline, train_data, labels, test_data, 
                                               gold, n_jobs, verbose, pre_dispatch, 
                                               rnd_seed=rnd_seed, test_name=test_name)
            else:
                scores = train_test_model(pipeline, train_data, labels, test_data, 
                                           gold,rnd_seed=rnd_seed, verbose=verbose, test_name=test_name)

            # create the return DataFrame
            model_name = cls.__class__.__name__.split('.')[-1]
            if model_name=='Pipeline':
                model_name = cls.steps[-1][0]
            df = create_scores_df(scores, model_name, test_name)

            # create a df with all the scores
            if type(ret) == pd.DataFrame:
                ret = ret.append(df)
            else:
                ret = df
    return ret


from time import time
from sklearn.model_selection import GridSearchCV


# first sover
logistic_params = {'C': [.3, .5, 1.0, 100, 1000],
 'class_weight': [None, 'balanced'],
 'dual': [False, True],
 'fit_intercept': [True, False],
 'intercept_scaling': [1],
 'max_iter': [100, 200, 500],
 'multi_class': ['ovr'],
 'penalty': ['l2'],
 'random_state': [rnd_seed],
 'solver': ['liblinear']}

# new solver
logistic_params_new = {'C': [.3, .5, 1.0, 100, 1000],
 'class_weight': [None, 'balanced'],
 'dual': [False],
 'fit_intercept': [True, False],
 'intercept_scaling': [1],
 'max_iter': [50, 100, 200, 500],
 'multi_class': ['ovr', 'multinomial'],
 'penalty': ['l2'],
 'random_state': [rnd_seed],
 'solver': ['newton-cg']}

# other solvers
logistic_params_sag = {'C': [.3, .5, 1.0, 100, 1000],
 'class_weight': [None, 'balanced'],
 'dual': [False],
 'fit_intercept': [True, False],
 'intercept_scaling': [1],
 'max_iter': [100, 200, 500, 5000],
 'multi_class': ['ovr', 'multinomial'],
 'penalty': ['l2'],
 'random_state': [rnd_seed],
 'solver': ['sag', 'lbfgs']}

def optimize_model(cls, params, train_data, labels, test_data, gold, scoring=scorer):
    """
    Given a model, a Grid Search is done to optimize the model
    """
    # train the model, cross validation of 10
    t0 = time()
    grid_search = GridSearchCV(cls, params, n_jobs=7, verbose=0, scoring=scoring, cv=10)
    grid_search.fit(train_data, labels)
    print("done in %0.3fs" % (time() - t0))
    
    # print the params
    print("Best score: %0.3f" % grid_search.best_score_)
    
    # check the best against the test data
    print("Test score:", scoring(grid_search.best_estimator_, test_data, gold))
    
    return grid_search

def print_params(model):
    from pprint import pprint
    print("Best parameters set:")
    best_parameters = model.best_params_
    pprint(best_parameters)

# -*- coding: utf-8 -*-
# The NRC script for 2013
# the original and modified version

# imports
import parser
from libs.parse import *
from libs.files import *
from libs.utils import *
from libs.resources import *
from collections import defaultdict
import argparse
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import traceback
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

tprint('Loading Lexicons')
nrc_emo = NRCEmotion()
nrc_sent = NRCSentiment140()
nrc_hash = NRCHashtagSentiment()
mpqa = MPQA()
bing = BingOpinion()
wc = WordCluster()

def main(args):

    # load the train dev and dist files
    tweets = []
    
    tprint("Loading training dataset...",)
    tr = TweetReader('../1-Input/tweeti-b.dist.full.tsv', task='B')
    tweets.extend(list(tr))
    tr = TweetReader('../1-Input/tweeti-b.dev.dist.full.tsv', task='B')
    tweets.extend(list(tr))
    tprint("done")
    tokens = get_tokens('../2-Processed/tweeti-b.dist.full.tagged.tsv')
    
    tprint("Loading training so_cal...")
    tr = TupleReader('../2-Processed/SO_CAL.full.train.tsv')
    so_cal = tr()
    tprint("done")

    tprint("Loading training SentiStrength...")
    tr = TripleReader('../2-Processed/SentiStrength.full.train.tsv')
    syn_strength= tr()
    tprint("done")
    
    tprint("Read the LIWC information")
    liwc = read_liwc('../2-Processed/liwc.tweeti-b.dev.dist.full.tsv')
    liwc.update(read_liwc('../2-Processed/liwc.tweeti-b.dist.full.tsv'))

    for tweet, token in zip(tweets, tokens):
        # make sure it is the same line
        if tweet['sid'] != token['sid']:
            tprint('Not the same ids')
            tprint('Tweet: {} ')
        tweet.update(token)

        # process so_cal
        if so_cal is not None:
            tweet['SO_CAL_VALUE'] = so_cal[tweet['sid']]

        # process syn strength
        if syn_strength:
            vals = syn_strength[tweet['sid']]
            tweet['SYN_STRENGTH_POS'] = vals[0]
            tweet['SYN_STRENGTH_NEG'] = vals[1]
            
        # check if should use liwc
        if args.liwc:
            tweet['LIWC'] = prefix_dict(liwc[int(tweet['sid'])], 'LIWC_')

    # create train features
    data_train = create_features(tweets, args)
    target = [encode_label.get(tweet['sentiment'], 0) for tweet in  tweets]

    # load the test files
    test_tweets = []
    tprint("Loading test dataset...")
    tr_test = TweetReader('../1-Input/twitter-test-input-B.tsv', task='B')
    test_tweets.extend(list(tr_test))
    tprint("done")
    tokens = get_tokens('../2-Processed/twitter-test-input-B.tagged.tsv')
    
    tprint("Loading test so_cal...")
    tr = TupleReader('../2-Processed/SO_CAL.full.test.tsv')
    so_cal = tr()
    tprint("done")

    tprint("Loading test SentiStrength...")
    tr = TripleReader('../2-Processed/SentiStrength.full.test.tsv')
    syn_strength= tr()
    tprint("done")
    
    tprint("Read the LIWC information")
    liwc_test = read_liwc('../2-Processed/liwc.twitter-test-input-B.tsv')

    for tweet, token in zip(test_tweets, tokens):
        # make sure it is the same line
        if tweet['sid'] != token['sid']:
            tprint('Not the same ids')
            tprint('Tweet: {} ')
        tweet.update(token)

        # process so_cal
        if so_cal is not None:
            tweet['SO_CAL_VALUE'] = so_cal[tweet['sid']]

        # process syn strength
        if syn_strength:
            vals = syn_strength[tweet['sid']]
            tweet['SYN_STRENGTH_POS'] = vals[0]
            tweet['SYN_STRENGTH_NEG'] = vals[1]
            
        # check if should use liwc
        if args.liwc:
            tweet['LIWC'] = prefix_dict(liwc_test[int(tweet['sid'])], 'LIWC_')

    # create features
    data_test = create_features(test_tweets, args)
    
    # tranform the list of dicts into feature vectors
    tprint("Vectorizing")
    vec = DictVectorizer()
    X = vec.fit_transform(data_train)
    y = np.array(target)
    X_test = vec.transform(data_test)
    
    # train and classify
    #clf = SGDClassifier(alpha=0.001, average=False, class_weight='auto', epsilon=0.1,
    #   eta0=0.0, fit_intercept=True, l1_ratio=0.5, learning_rate='optimal',
    #   loss='hinge', n_iter=30, n_jobs=1, penalty='elasticnet',
    #   power_t=0.5, random_state=None, shuffle=True, verbose=0,
    #   warm_start=False)
    clf = LinearSVC()
    from sklearn.cross_validation import StratifiedKFold
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')
    rfecv.fit(X, y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    if(args.featureSelection):
        classif =  f_classif
        sel = SelectPercentile(classif, percentile=95)
        X_train = sel.fit_transform(X,y)
        X_test_final = sel.transform(X_test)
    else:
        X_final = X
        X_test_final = X_test
    
    tprint("Training the classifier")
    clf.fit(X_final, y)
    
    # predict
    tprint("Predicting")
    pred = clf.predict(X_test_final)
    
    # output 
    tprint("Saving to file...")
    output_filename = '../3-Output/pred_NRC2013.tsv'
    with codecs.open(output_filename, 'w', 'utf8') as f:
        for tweet, label in zip(test_tweets, pred):
            line = '\t'.join([tweet["sid"], tweet["uid"], decode_label.get(label, 'neutral'),
                              tweet["text"]]) + '\n'
            f.write(line)
    tprint("done")
    

def get_tokens(file):
    '''
    Load tokens and pos created previously from ARk tokenizer
    '''
    info = TabReader(file)

    data = []
    for l in info.iterlines():
        # get the line sid
        line = {}
        line['sid'] = l[0]
        line['tokens'] = []
        pos_dict = defaultdict(int)
    
        # get the tokens and their pos
        for val in l[1:]:
            token, pos = val.split(' ')
        
            # save values
            line['tokens'].append(token)
            pos_dict[pos] += 1
        
        line['pos'] = pos_dict
        data.append(line)  
    
    return data

def create_features(tweets, flags):
    '''
    Create the features according to the paper of NRC 2013
    '''
    count = 0
    pos = []
    data = []
    # wll tokenize first and then apply the POS for all.
    #tprint("Tokenizing tweets.",)
    #t = [ tokenize(tweet['text']) for tweet in tweets]
    #if(flags.count_pos):
    #    pos = nltk.pos_tag_sents(t)
    
    for tweet in tweets:
        # process each tweet individually
        tweet['tokens'] = normalize(tweet['tokens'])
        try:
            data.append(create_tweet_features(tweet, flags))
        except:
            tprint('Error in tweet nr {}'.format(count))
            print_exception() 
            tprint(tweet)
        #data.append(create_lexicon_features(tweet['tokens']))
        # progress
        count += 1
        if (count%500)==0:
            tprint('Done %d' %count)
            
    tprint(tweets[0])
    tprint('1 line of info {}'.format( sorted(data[0].iteritems())))
    
    return data

def create_tweet_features(tweet, flags):
    '''
    Process each tweet, one at a time creating word related features
    '''
    # structures
    #import pudb
    #pudb.set_trace()
    fdict = defaultdict(int)
    data = []
    tokens = tweet['tokens']
    
    uni_lex = {}
    bi_lex = {}
    pairs_lex = {}
    
    
    # normalize hash, url and mentions from text
    text = ' '.join( normalize(tweet['text'].lower().split(' ')))
    #text = tweet['text']
    
    # check the negation
    neg = False
    new_tokens = []
    over_polarity = []
    neg_num = 0
    for i, token in enumerate(tokens):
        # normalize
        token_norm = token
        
        # find the lexicon values
        polarity = {}
        polarity.update(nrc_emo.get_polarity(token_norm))
        polarity.update(nrc_sent.get_polarity(token_norm))
        polarity.update(mpqa.get_polarity(token_norm))
        polarity.update(bing.get_polarity(token_norm))
        
        # save the new token 
        if neg:
            token_norm = token_norm + '_NEG'
            
            n = {}
            for k,v in polarity.iteritems():  
                n[k+'_NEG'] = v
            polarity = n

            neg_num += 1
            
        new_tokens.append(token_norm)
        over_polarity.append(polarity)
        
        # check if this is a negated word
        if not neg and negated_token_regex.search(token):
            neg = True
        
        # check if the end of negated sentence
        if neg and end_negated_regex.search(token):
            neg = False
    
    # create statistics
    if len(over_polarity)>0:
        df_pol = pd.DataFrame(over_polarity)
        df_pol.fillna(0, inplace=True)
    
        # count statistics
        stats_polarity = {}
        tmp=df_pol[df_pol>0].count(axis=0)
        stats_polarity.update(suffix_dict(tmp, '_COUNT'))
        
        # max statistics
        tmp=df_pol.max().to_dict()
        stats_polarity.update(suffix_dict(tmp, '_MAX'))
        
        # sum
        tmp=df_pol.sum().to_dict()
        stats_polarity.update(suffix_dict(tmp, '_SUM'))
        
        # last token
        stats_polarity.update(suffix_dict(over_polarity[-1], '_LAST'))
    
    # create the ngrams
    if(flags.ngrams):
        #tprint('Creating ngrams')
        #tokens = tokenize(tweet)
        bi_grams = bigrams(new_tokens)
        tri_grams = trigrams(new_tokens)
        four_grams = ngrams(new_tokens, 4)
        
        # create non contiguous sequences
        non_bi = [item for gr in bi_grams for item in non_contiguous(list(gr))]
        non_tri = [item for gr in tri_grams for item in non_contiguous(list(gr))]
        non_four = [item for gr in four_grams for item in non_contiguous(list(gr))]
        
        # concatenate all
        data = bi_grams + tri_grams + four_grams + non_bi + non_tri + non_four
        
        # transform all new_tokens into strings
        data = [' '.join(tok) for tok in data]
        data += new_tokens
        
        # check if it should generate lexicon info 
        if flags.lexicon:
            #uni_lex = create_lexicon_features(token)
            bi_grams = bigrams(token)
            bi_lex = create_lexicon_features([' '.join(tok) for tok in bi_grams], 
                                             'bigram')
            four_grams = ngrams(new_tokens, 4)
            pairs_lex = create_lexicon_features([' '.join(tok).replace('*','---') 
                                             for tok in bi_grams], 
                                            'pairs')

    # create char ngrams
    if flags.chargrams:
        #tprint('Creating chargrams')
        data += char_ngrams(text,3)
        data += char_ngrams(text,4)
        data += char_ngrams(text,5)
        
    # transform all to dict
    for t in data:
        fdict[convert_token(t)] += 1
        
    # add the lexicons
    fdict.update(stats_polarity)
    fdict.update(bi_lex)
    fdict.update(pairs_lex)
    fdict['NEGATED_NUM'] = neg_num

        
    # count the number of caps
    if(flags.count_caps):
        #tprint('Creating all caps')
        fdict['NR_ALL_CAPS'] = count_all_caps(token)
    
    # count the number of caps
    if(flags.count_mentions):
        #tprint('Creating count of mentions')
        fdict['NR_MENTIONS'] = count_mentions(token)
        
    # count the number of hash
    if(flags.count_hash):
        #tprint('Number of hash')
        fdict['NR_HASH'] = count_hash(token)
        
    # cout number of pos
    if(flags.count_pos):
    #    #tprint('Creating POS count')
        fdict.update(prefix_dict(tweet['pos'], 'POS_'))
        
    # punctuation contiguous
    if(flags.punctuation):
        #tprint('Finding elongated words')
        elong = find_elongated_punct(tokens)
        fdict['PCT_ELONG'] = len(elong)
        
        # check the last token
        last = token[-1]
        if last.find('!') >= 0 or last.find('?') >= 0:
            fdict['PCT_LAST!?'] = True
        else:
            fdict['PCT_LAST!?'] = False
    
    # polarity emoticons 
    if(flags.emoticon):
        # check if there is a positive or negative emoticon
        hasPos = False
        hasNeg = False
        for tok in tokens:
            if tok.find('http')>=0 or time_regex.search(tok) or \
               tok.find('.com')>=0:
                continue
            if emoticon_pos_regex.search(tok):
                #tprint('POS ' + tok
                hasPos = True
            if emoticon_neg_regex.search(tok):
                #tprint('NEG ' + tok
                hasNeg = True
        fdict['EMOT_POS'] = hasPos
        fdict['EMOT_NEG'] = hasNeg
        
        # check if the last token is a emoticon positive or negative
        last = tokens[-1]
        if emoticon_pos_regex.search(last):
            fdict['EMOT_LAST'] = 1
        elif emoticon_neg_regex.search(last):
            fdict['EMOT_LAST'] = -1
        else:
            fdict['EMOT_LAST'] = 0
    
    # check if there is elongated words
    if(flags.elongated):
        #tprint('Finding elongated words')
        elong = find_elongated(tokens)
        fdict['FLG_ELONGATED'] = len(elong)
        
    # cluster
    if(flags.cluster):
        clust = defaultdict(bool)
        for tok in tokens:
            cluster = wc.get_cluster(tok)
            if cluster:
                clust['CLUST_' + cluster] = True
            
        fdict.update(clust)
    
    # emotions
    # process so_cal
    if flags.so_cal is not None:
        fdict['SO_CAL_VALUE'] = tweet['SO_CAL_VALUE'] 

    # process syn strength
    if flags.syn_strength:
        fdict['SYN_STRENGTH_POS'] = tweet['SYN_STRENGTH_POS']
        fdict['SYN_STRENGTH_NEG'] = tweet['SYN_STRENGTH_POS']
    
    # process LIWC
    if flags.liwc:
        fdict.update(tweet['LIWC'])
        
    return fdict
    
def normalize(tokens):
    '''
    Normalize a set of tokens of a twitter
    Params:
        list of tokens
    Returns:
        normalized tokens
    '''
    data = None
    # remove url, hash and mentions
    data = normalize_mentions(tokens)
    #data = normalize_hash(data)
    data = normalize_url(data)
    
    # 
    if type(tokens)==list:
        data = [token.lower() for token in tokens]

    return data
    
def create_lexicon_features(tokens, ptype=None):
    '''
    Create features related to the lexicons
    Params:
        list of tokens
    Returns:
        list of the features related to the lexicons
    '''
    data = {}
    lex = LexiconStats(nrc_emo, 'NRCEMO')
    data.update(lex.get_stats(tokens))
    
    lex = LexiconStats(nrc_sent, 'NRCSENT', ptype=ptype)
    data.update(lex.get_stats(tokens))
    
    lex = LexiconStats(nrc_hash, 'NRCHASH', ptype=ptype)
    data.update(lex.get_stats(tokens))
    
    lex = LexiconStats(mpqa, 'MPQA')
    data.update(lex.get_stats(tokens))
    
    lex = LexiconStats(bing, 'BING')
    data.update(lex.get_stats(tokens))
    
    return data


# the parser that will check params
def create_parser():
    parser = argparse.ArgumentParser(description='Run the prediction using NRC and features')
    parser.add_argument('-ng','--ngrams', 
                       help='Should generate ngrams', default=True)
    parser.add_argument('-cg','--chargrams',
                       help='Should generate chargrams', default=True)
    parser.add_argument('-cc','--count_caps', 
                       help='Count number of all cap letters words', default=True)
    parser.add_argument('-cm','--count_mentions', default=True,
                       help='Count number of mentions')
    parser.add_argument('-ch','--count_hash', default=True,
                       help='Count number of hash words')
    parser.add_argument('-cp','--count_pos', default=True,
                       help='Count number of words per pos')
    parser.add_argument('-el','--elongated', default=True,
                       help='Count elongated words')    

    parser.add_argument('-le','--lexicon', 
                       help='Use lexicons BING, MPQA, NRC', default=True)
    parser.add_argument('-cl','--cluster', default=True,
                       help='Use word cluster')
    parser.add_argument('-pu','--punctuation', default=True,
                       help='Create punctuation features')
    parser.add_argument('-em','--emoticon', default=True,
                       help='Count number of words per pos')

    parser.add_argument('-so','--so_cal', default=True,
                       help='Create SO_CAL Features') 
    parser.add_argument('-ss','--syn_strength', default=True,
                       help='Use syn strength features') 
    parser.add_argument('-lw','--liwc', default=False,
                       help='Use LIWC features') 
                     
    return parser

args = Map(dict(chargrams=True, count_caps=True, count_hash=True, 
             count_mentions=True, count_pos=True, elongated=True, ngrams=True,
             lexicon = True, cluster = True, punctuation=True, emoticon=True,
             so_cal=True, syn_strength=True, liwc=True))
#tprint(args)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    tprint(args)
    main(args)
import libs.resources as r
from libs.tweet import Tweet
import libs.files as fh
import libs.parse as p
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier
import codecs
from os import path
from libs.db import TweetsDB, RunWrapper
from sklearn.metrics import f1_score, classification, classification_report
from datetime import datetime
from sklearn.externals import joblib
import argparse
import sys

parser_descr = """Analyse tweets and create sentiment analysis predictions \n
command line:
python awa_sent_analyser.py -c config.yaml\n\n

Ex. config file, lexicons that are not listed and set to false will be processed: \n
lexicons:
    wna: False
    tslex: True
output: '../3-Output/all.csv'
file_type: B
train_file_name: "../1-Input/twitter-train-cleansed-B.csv"
test_file_name: "../1-Input/twitter-test-input-B.tsv" """

def get_lexicons_features(tweet, cfg):
    """
    Create lexicons' features according to config files.The lexicons should be inside
    lexicons key. Non existant lexicons will be treated as TRUE
    """
    features = {}
    dic_tokens = {}
    # process all available dictionaries, unless discarded in config file
    for lex in r.lexs:
        # check if discarded
        if cfg['lexicons'].get(lex.prefix.lower(), True):
            #print lex.prefix
            ret, tl = lex.process_tweet(tweet)
            #print ret
            if ret:
                features.update(ret)
                dic_tokens[lex.prefix] = tl

    return features, dic_tokens


def create_features(file_name, file_type, cfg, db):
    """
    Create the features according to config files. 
    """
    features_tweets = []
    sentiment_tweets = []
    tweets = []
    tr = fh.TweetReader(file_name, file_type)

    for t in tr:
        # load the tweet
        tw = Tweet(t["text"])
        sentiment_tweets.append(p.encode_label.get(t["sentiment"], None))
        tw.process()
        # import pudb;pudb.set_trace()
        features = {}

        # create the words features
        tweets_feat = tw.get_all_features()

        # use the lexicons to process the tweets
        lex_features, lex_tokens = get_lexicons_features(tw, cfg)
        
        # append the created features to the list
        features.update(tweets_feat)
        features.update(lex_features)
        features_tweets.append(features)

        # save the info to db
        d = {}
        # d['tweet_features'] = tweets_feat
        # d['lex_features'] = lex_features
        # d['lex_tokens'] = lex_tokens
        d['tweet'] = tw.__dict__
        d['sid'] = t["sid"]
        d['uid'] = t["uid"]
        d['sentiment'] = t["sentiment"]
        db.save_tweet(d)
        
        # save the tweets properties for later
        tweets.append(tw)
        
    return features_tweets, sentiment_tweets


def main(args):
    """
    Process the tweets, returning an output file
    """
    # read the config file
    cfg = fh.read_config_file(args.cfg)
    file_type = cfg['file_type']

    # check if should produce a report
    report = cfg.get('report_file', None)
    report_f = None
    if report:
        # open the file
        report_f = open(report, 'w')

    # open the TweetsDB connections
    db_train = TweetsDB('sa_train', drop_db=True)
    db_test = TweetsDB('sa_test')

    # read the train file
    print "Processing train file",
    train_file_name = cfg['train_file_name']
    train_feat, labels = create_features(train_file_name, file_type, cfg, db_train)

    # save the dictionary info to db
    tr_stats = r.get_lexs_stats()
    r.reset_lexs_stats()
    print "done"

    if report:
        print >> report_f, r.pd.DataFrame(tr_stats)

    # read the test file
    print "Processing test file",
    test_file_name = cfg['test_file_name']
    test_feat, gold = create_features(test_file_name, file_type, cfg, db_test)

    # get the test stats
    ts_stats = r.get_lexs_stats()
    print "done"

    # create the feature vector
    print "Training model",
    vec = DictVectorizer()
    X = vec.fit_transform(train_feat)
    y = np.array(labels)
    X_test = vec.transform(test_feat)

    
    # train the model
    clf = SGDClassifier(penalty='elasticnet', alpha=0.0001, l1_ratio=0.85,
                        n_iter=1000, n_jobs=-1)
    clf.fit(X, y)
    print "done"

    print "Predicting",
    pred = clf.predict(X_test)
    print "done"

    # calculate score
    print "Saving information to db",
    score = f1_score(gold, pred, labels=[-1, 1], average='macro')
    prfs = classification.precision_recall_fscore_support(gold, pred)

    # save run time 
    run = RunWrapper(pred, gold, score, tr_stats, ts_stats, prfs, clf)
    run_info = run.save()
    print "done"

    # save model
    print "Saving model to file",
    date = run_info['date'].strftime('%Y%m%d.%H%M%S')
    model = run_info['model'].split('.')[-1]
    file_name = '{}.{}.pkl'.format(date, model)
    joblib.dump(clf, path.join(cfg['model_out_dir'], file_name))
    print "done"

    # save the predicted values
    tr = fh.TweetReader(test_file_name, file_type)
    output_filename = cfg['output']
    print "Saving output to file,", output_filename, 
    with codecs.open(output_filename, 'w', 'utf8') as out:
        for tweet, label in zip(tr, pred):
            line = '\t'.join([tweet["sid"], tweet["uid"],
                              p.decode_label[label], tweet["text"]]) + '\n'
            out.write(line)

            db_test.save_tweet_pred_sent(tweet["sid"], p.decode_label[label])

    print "done"
    print "\n\nRun the following code on shell"
    print "python scorer.py b %s ../1-Input/twitter-test-GOLD-B.tsv"  % output_filename


def create_parser():
    """
    Creates a parser to help with the inputs
    """
    parser = argparse.ArgumentParser(description=parser_descr, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c','--cfg',
                       help="""Pass a config file with the configurations for the parser.\n""")

    return parser

if __name__ == '__main__':
    # get the params for the process
    parser = create_parser()
    args = parser.parse_args()

    # return if no params found
    if not args.cfg:
        print "Please check the help with 'python %s -h'" %__file__
        sys.exit(0)


    main(args)
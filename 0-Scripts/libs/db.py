# -*- coding: utf-8 -*-
#
# Handles TweetsDB information insertion
# Used to better check the information created
# 
#
from pymongo import MongoClient, errors
from datetime import datetime
from utils import change_key_recursively

class TweetsDB(object):
    """Inits the database, recreate if needed"""
    def __init__(self, col_prefix=None, db_name='sent_analysis', drop_db=False):
        super(TweetsDB, self).__init__()
        if not col_prefix:
            n = datetime.now()
            col_prefix = '%s%s%s'%(n.day,n.hour,n.minute)
        self.col_prefix = col_prefix
        self.db_name = db_name
        self.client = MongoClient()
        self.db = self.client[db_name]

        # check if the connection worked ok. if not, all  the other operations
        # should not work
        try:
            print self.client.server_info() 
            self.connected = True

        except errors.ServerSelectionTimeoutError as err:
            # do whatever you need
            self.connected = False

        # init procedure
        if drop_db and self.connected:
            self.client.drop_database(db_name)

    def save_in_db(self, dic, collection):
        """
        Inserts a dictionary into the db
        """
        if not self.connected:
            False

        # check if we are talking about a different collection
        collection = self.col_prefix + '_' + collection

        # insert the information and return properly
        col = self.db[collection]
        
        try:
            ret = col.insert_one(dic)
            return None
        except:
            # import pudb
            # pudb.set_trace()
            nd = change_key_recursively(dic)
            try:
                ret = col.insert_one(nd, bypass_document_validation=False)
            except:
                print 'Tweet Error', dic['uid']

    def save_tweet(self, tweet):
        """
        Saves a tweet class specifically
        """
        if not self.connected:
            False

        # create a dictionary over the tweet class
        ser = tweet
        
        # save into the db
        self.save_in_db(ser, 'tweets')
        
    def save_tweet_pred_sent(self, sid, sentiment, run=0):
        """
        Save the predicted sentiment
        """
        if not self.connected:
            False
            
        # check if we are talking about a different collection
        collection = self.col_prefix + '_tweets'

        # insert the information and return properly
        col = self.db[collection]
        
        return col.update_one({'sid': sid}, {'$set': {'pred_sent_{}'.format(run): sentiment}})


class RunWrapper(object):
    """docstring for RunWrapper"""
    def __init__(self, pred, gold, score, train_stats, test_stats,
                 precision_recall_fscore_support, model, db_name='sent_analysis', 
                 collection='runs'):
        super(RunWrapper, self).__init__()
        self.pred = pred
        self.gold = gold
        self.score = score
        self.train_stats = train_stats
        self.test_stats = test_stats
        self.precision_recall_fscore_support = precision_recall_fscore_support
        self.model = model

        # connect to the db
        self.client = MongoClient()
        self.db = self.client[db_name]
        self.collection = collection


    def save(self):
        """
        Given a previous information related to this run, save it into
        the db
        """
        cur_date = datetime.now()

        # transform the precision recall to list
        p = (self.precision_recall_fscore_support[0].tolist(),
             self.precision_recall_fscore_support[1].tolist(),
             self.precision_recall_fscore_support[2].tolist(),
             self.precision_recall_fscore_support[3].tolist()
            )

        # create dictionary to save into db
        dic = {
            'train_stats': self.train_stats,
            'test_stats': self.test_stats,
            'score': self.score,
            'precision_recall_fscore_support': p,
            'gold': self.gold,
            'pred': self.pred.tolist(),
            'date': cur_date,
            'model': self.model.__class__.__module__
        }

        # save to the run collection
        col = self.db[self.collection]
        
        #try:
        ret = col.insert_one(dic)
        return dic
        #except errors.InvalidDocument:
            # nd = self._change_key_recursively(dic)
        #    ret = col.insert_one(nd, bypass_document_validation=False)
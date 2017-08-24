# -*- coding: utf-8 -*-
#Original script
# Copyright (c) 2013 Tobias Günther and Lenz Furrer.
# All rights reserved.
#
# The class TweetReader
# adapted to my structure. Please refer to:
#    PDF: http://aclweb.org/anthology/S/S13/S13-2054.pdf
#    BibTex: http://aclweb.org/anthology/S/S13/S13-2054.bib
#

#
# Author: Adriano W Almeida
# date: 2015
#
import codecs
import pandas as pd
import yaml
from HTMLParser import HTMLParser

unescape = HTMLParser().unescape

class TweetReader(object):
    def __init__(self, filename, task):
        self.filename = filename

        if task not in ["A", "B", "2016A"]:
            raise ValueError("Task must be 'A' or 'B'.")
        self.task = task

    def __iter__(self):
        with codecs.open(self.filename, 'r', encoding='utf-8', errors='ignore') as f:
                i = 0
                for i, line in enumerate(f):

                    line = line.strip('\r\n')
                    fields = line.split('\t')

                    if self.task == "A" and len(fields) == 6:
                        sid, uid, swp, ewp, senti, text = fields
                        if text != "Not Available":
                            yield {
                                "sid": sid,
                                "uid": uid,
                                "start_pos": int(swp),
                                "end_pos": int(ewp),
                                "sentiment": senti,
                                "text": text
                            }
                    elif self.task == "B" and len(fields) == 4:
                        sid, uid, senti, text = fields

                        if text != "Not Available":
                            senti = senti.strip('"')
                            if senti not in ('positive', 'negative', 'neutral'):
                                senti = 'neutral'
                            yield {
                                "sid": sid,
                                "uid": uid,
                                "sentiment": senti.strip('"'),
                                "text": unescape(text)
                            }
                    elif self.task == "2016A" and len(fields) == 3:
                        sid, senti, text = fields
                        if text != "Not Available":
                            yield {
                                "sid": sid,
                                "sentiment": senti.strip('"'),
                                "text": text
                            }
                    else:
                        error = "Malformed line: %s (too many/few tabs)" % (i + 1)
                        print(i, line)
                        #raise ValueError(error)

                print("Read {} rows from {}".format(i, self.filename))

    def iterator(self):
        return self.__iter__()


class SynReader(object):
    def __init__(self, filename):
        self.filename = filename

        #if task not in ["A", "B"]:
        #    raise ValueError("Task must be 'A' or 'B'.")
        #self.task = task

    def __iter__(self):
        with codecs.open(self.filename, 'r', 'utf8') as f:
            for i, line in enumerate(f):

                line = line.strip('\n')
                fields = line.split('\t')
                if len(fields) == 8:
                    sid, h, sd, a, f, d, su, v = fields
                    yield {
                        "sid": sid,
                        "syn_happy": h,
                        "syn_sadness": sd,
                        "syn_anger": a,
                        "syn_fear": f,
                        "syn_disgust":d,
                        "syn_surprise":su,
                        "syn_valence":v
                    }
                else:
                    error = "Malformed line: %s (too many/few tabs)" % (i + 1)
                    raise ValueError(error)


class TupleReader(object):
    def __init__(self, filename):
        self.filename = filename

        #if task not in ["A", "B"]:
        #    raise ValueError("Task must be 'A' or 'B'.")
        #self.task = task

    def __call__(self):
        with codecs.open(self.filename, 'r', 'utf8') as f:
            ret = {}
            for i, line in enumerate(f):

                line = line.strip('\n')
                fields = line.split('\t')
                if len(fields) == 2:
                    sid, value = fields
                    ret[sid] = value
                else:
                    error = "Malformed line: %s (too many/few tabs)" % (i + 1)
                    raise ValueError(error)
        return ret


class TripleReader(object):
    def __init__(self, filename):
        self.filename = filename

        #if task not in ["A", "B"]:
        #    raise ValueError("Task must be 'A' or 'B'.")
        #self.task = task

    def __call__(self):
        with codecs.open(self.filename, 'r', 'utf8') as f:
            ret = {}
            for i, line in enumerate(f):

                line = line.strip('\n')
                fields = line.split('\t')
                if len(fields) == 3:
                    sid, value1, value2 = fields
                    ret[sid] = (value1, value2)
                else:
                    error = "Malformed line: %s (too many/few tabs)" % (i + 1)
                    raise ValueError(error)
        return ret


class TabReader(object):
    def __init__(self, filename, offset=0):
        self.filename = filename
        self.offset = offset

        #if task not in ["A", "B"]:
        #    raise ValueError("Task must be 'A' or 'B'.")
        #self.task = task

    def __call__(self, fieldNum=None):
        '''
        Read the information.
        Params:
            fieldNum: number of columns in the file.
        Returns:
            Dict(ids=(all,the,fields))
        '''
        with codecs.open(self.filename, 'r', 'utf8') as f:
            ret = {}
            for _ in range(self.offset):
                _ = f.next()

            for i, line in enumerate(f):
                line = line.strip('\n')
                fields = line.split('\t')
                if len(fields) == fieldNum or fieldNum is None:
                    sid = fields[0]
                    ret[sid] = tuple(fields[1:])
                else:
                    error = "Malformed line: %s (too many/few tabs)" % (i + 1)
                    raise ValueError(error)
        return ret
        
    def iterlines(self):
        '''
        Create a iterator with the lines.
        returns:
            each line as a list of fields
        '''
        with codecs.open(self.filename, 'r', 'utf8') as f:
            ret = {}
            for _ in range(self.offset):
                _ = f.next()

            for i, line in enumerate(f):
                line = line.strip('\n')
                yield line.split('\t')


def read_liwc(file_name):
    '''
    read a LIWC processed file and return a dict with a map with the sid as key
    '''
    # read the file discard the last column that does not represent anything
    data = pd.read_csv(file_name, sep='\t')
    del data[data.columns[-1]]
    
    # set the sid as index return as a dict
    return data.set_index('sid').T.to_dict()


def read_config_file(file_name):
    """
    Read the config file 
    """
    # read the yaml file
    cfg = {}
    with open(file_name, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    # check if the retquired fields are there
    req = ['output', 'lexicons']
    for k in req:
        if k not in cfg.keys():
            raise Exception('Required keys not found', k)
    
    return cfg


# simple file reader version
# from HTMLParser import HTMLParser
# import codecs
# unescape = HTMLParser().unescape

# def process_tweet(line):
#     fields = line.strip("\r\n").split('\t')
#     sent = fields[2].replace('"', '')
    
#     return dict(
#         sid = int(fields[0]),
#         uid = int(fields[1]),
#         sent = p.encode_label[sent],
#         msg = unescape(fields[3]),
#     )
# def read_file(file_name):
#     with codecs.open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
#         info = f.readlines()
        
#     data = []
#     for line in info:
#         tw = process_tweet(line)
#         if tw['msg'] == 'Not Available':
#             continue
#         data.append(tw)
#     labels = [ tw['sent'] for tw in data]
#     msg = [ tw['msg'] for tw in data]
    
#     return data, labels, msg
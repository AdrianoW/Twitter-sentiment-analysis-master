# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import traceback
from subprocess import Popen, PIPE


def tprint(*msg):
    '''
    Print the message with a time before iter
    '''
    print('%s -' % time.strftime('%X %x %Z'), *msg)


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
        
def print_exception():
    formatted_lines = traceback.format_exc().splitlines()
    for i in range(-5,0):
        tprint(formatted_lines[i])
    
def suffix_dict(d, suffix):
    """
    Add a suffix to all the keys of the dict
    Args:
        d: dictionary
        suffix: the suffiy to append the keys to

    Returns: dict with the keys appended with suffix

    """
    tmp = {}
    for k,v in d.iteritems():
        tmp[k+suffix] = v
    return tmp


def prefix_dict(d, prefix):
    """
    Add a prefix to all the keys of the dict
    Args:
        d: dictionary
        prefix: the suffiy to append the keys to

    Returns: dict with the prefix added to keys

    """
    tmp = {}
    for k,v in d.iteritems():
        tmp[prefix+k] = v
    return tmp


def jar_wrapper(args):
    '''
    Wrapper to call a java script and get the results back
    params:
        args: list of params for the Popen

    ex:
        jar_wrapper(['java.jar'])
    '''
    popen_args = ['java', '-jar']
    process = Popen(popen_args + args, stdout=PIPE, stderr=PIPE)
    ret = []
    while process.poll() is None:
        line = process.stdout.readline()
        if line != '' and line.endswith('\n'):
            ret.append(line[:-1])
    stdout, stderr = process.communicate()
    ret += stdout.split('\n')
    if stderr != '':
        ret += stderr.split('\n')
    ret.remove('')
    return ret

def change_key_recursively(search_dict):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.iteritems():

        if key[0]=='$' or '.' in key:
            nk = '_'+key.replace('.','PP')
            search_dict[nk] = search_dict[key]
            del search_dict[key]
            key = nk

        if isinstance(value, dict):
            search_dict[key] = change_key_recursively(value)

        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    more_results = change_key_recursively(item)
                    new_list.append(more_results)
                else:
                    new_list.append(item)
            search_dict[key] = new_list

    return search_dict

def get_sent_idx(labels):
    """
    Returns the index to positive, neutral and negative tweets

    Params:
        train_data: list of tweets
        labels: the labels associated with the tweets (1,0, -1)

    ret:
        ([positive], [neutral], [negative]) indexes

    """
    pos = []
    neg = []
    neu = []

    # get the indexes
    pos = [ i for i, l in enumerate(labels) if l ==1]
    neg = [ i for i, l in enumerate(labels) if l ==-1]
    neu = [ i for i, l in enumerate(labels) if l ==0]
    return pos, neu, neg

from nltk.corpus import wordnet as wn

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return ''

def penn_to_lew(tag):
    if is_adjective(tag):
        return u'ADJECTIVE'
    elif is_noun(tag):
        return u'NOUN'
    elif is_adverb(tag):
        return u'ADVERB'
    elif is_verb(tag):
        return u'VERB'
    return ''

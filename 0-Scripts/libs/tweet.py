# -*- coding: utf-8 -*-
# Holds a tweet and its processed variations
# token by token
# brigrams, trigrams, etc
import parse as p

class Tweet(object):
    """Holds a tweet and the processed tokens"""
    def __init__(self, msg, sent=None, sid=0, uid=0):
        super(Tweet, self).__init__()

        # original message, cleaned message
        self.msg = msg
        self.msg_cleaned = None
        self.sent = sent
        self.sid = sid
        self.uid = uid

        # processed message and different tokens
        self.tokens = None
        self.bigrams = None
        self.trigrams = None
        self.ngrams = None
        self.non_contiguous = None
        
        # additional features
        self.all_caps_num = 0
        self.mentions_num = 0
        self.hash_num = 0
        self.pos_num_dic = None

        # found stats 
        self.found_tokens_by_lex = {}
        self.found_tokens = {}
        self.found_tokens_total = 0

        # bigrams found stats 
        self.found_bigrams_by_lex = {}
        self.found_bigrams = {}
        self.found_bigrams_total = 0

        # found trigrams stats
        self.found_trigrams_by_lex = {}
        self.found_trigrams = {}
        self.found_trigrams_total = 0

        # found non contiguous stats
        self.found_non_contiguous_by_lex = {}
        self.found_non_contiguous = {}
        self.found_non_contiguous_total = 0

        # final created features
        self.final_lex_features = {}

        # the matching between lexicon and features
        self.lex_sent_match = {}

        
    def __repr__(self):
        # print general info
        ret = 'uid, sid: {},{}\n'.format(self.uid, self.sid)
        ret += 'Sentiment: {}'.format(self.sent)
        ret +=  "msg: " + self.msg + "\n"

        # print token stuff
        if self.tokens:
            ret = ret + "\n\ntokens: " + "|".join(self.tokens)
        if self.bigrams:
            ret = ret + "\n\nbrigrams: " + "|".join(self.bigrams)
        if self.trigrams:
            ret = ret + "\n\ntrigrams: " + "|".join(self.trigrams)
        if self.non_contiguous:
            ret = ret + "\n\nnon_contiguous: " + "|".join(self.non_contiguous)

        return ret 


    def process(self, token_type='informal', cfg={}):
        """ 
        Creates tokens, brigrams, trigrams
        """
        # use ntlk to make the ngrams
        self.tokens = self._tokenize_clean(token_type, cfg)
        self.bigrams = [ " ".join(t)  for t in p.bigrams(self.tokens) ]
        self.trigrams = [ " ".join(t)  for t in p.trigrams(self.tokens) ]
        self.non_contiguous = p.non_contiguous(self.tokens)
        
    def _tokenize_clean(self, token_type, cfg):
        """
        Tokenize and clean the tweet
        """
        tokens = p.tokenize(self.msg, token_type)
        
        # count the original data
        self.all_caps_num = p.count_all_caps(tokens)
        self.mentions_num = p.count_mentions(tokens)
        self.hash_num = p.count_hash(tokens)
        #self.pos_num_dic = p.count_pos(self.msg) 

        # remove stop
        if cfg.get('pre_process_remove_stop', True):
            tokens = p.remove_stop(tokens)     
        
        # normalize data according to the config
        if cfg.get('pre_process_lower', True):
            tokens = p.normalize_lower(tokens)
        if cfg.get('pre_process_mentions', True):
            tokens = p.normalize_mentions(tokens)
        if cfg.get('pre_process_hash', True):
            tokens = p.normalize_hash(tokens)
        if cfg.get('pre_process_url', True):
            tokens = p.normalize_url(tokens)
        if cfg.get('pre_process_number', True):
            tokens = p.normalize_number(tokens)   
        
        # save the normalized message
        self.msg_cleaned = " ".join(tokens)
        
        # TODO: create negated tokens
        
        # finally return the data
        return tokens

    def save_tokens_found(self, lex, tokens):
        """
        Save the tokens that were found in each dictionary
        """
        # get the lexicon info if it exists
        tmp = self.found_tokens_by_lex.get(lex, {})
        tmp_tokens = tmp.get('tokens', [])
        tmp_total = tmp.get('total', 0)
    
        # update info
        tmp_tokens += tokens
        tmp_total += len(tokens)

        # save it again
        tmp['tokens'] = tmp_tokens
        tmp['total'] = tmp_total
        self.found_tokens_by_lex[lex] = tmp

        # for each token, save what was the value given by a dictionary
        for tok_val in tokens:
            # get the token current value in case there is
            token, val = tok_val.items()[0]
            t_dic = self.found_tokens.get(token, {})

            # add the values of this lexicon to this token and save it
            t_dic[lex] = val
            self.found_tokens[token] = t_dic

        # refresh the total tokens
        self.found_tokens_total = len(self.found_tokens.keys())

    def save_bigrams_found(self, lex, bigrams):
        """
        Save the bigrams that were found in each dictionary
        """
        # get the lexicon info if it exists
        tmp = self.found_bigrams_by_lex.get(lex, {})
        tmp_bigrams = tmp.get('bigrams', [])
        tmp_total = tmp.get('total', 0)
    
        # update info
        tmp_bigrams += bigrams
        tmp_total += len(bigrams)

        # save it again
        tmp['bigrams'] = tmp_bigrams
        tmp['total'] = tmp_total
        self.found_bigrams_by_lex[lex] = tmp

        # for each token, save what was the value given by a dictionary
        for tok_val in bigrams:
            # get the token current value in case there is
            token, val = tok_val.items()[0]
            t_dic = self.found_bigrams.get(token, {})

            # add the values of this lexicon to this token and save it
            t_dic[lex] = val
            self.found_bigrams[token] = t_dic

        # refresh the total bigrams
        self.found_bigrams_total = len(self.found_bigrams.keys())

    def save_trigrams_found(self, lex, trigrams):
        """
        Save the trigrams that were found in each dictionary
        """
        # get the lexicon info if it exists
        tmp = self.found_trigrams_by_lex.get(lex, {})
        tmp_trigrams = tmp.get('trigrams', [])
        tmp_total = tmp.get('total', 0)
    
        # update info
        tmp_trigrams += trigrams
        tmp_total += len(trigrams)

        # save it again
        tmp['trigrams'] = tmp_trigrams
        tmp['total'] = tmp_total
        self.found_trigrams_by_lex[lex] = tmp

        # for each token, save what was the value given by a dictionary
        for tok_val in trigrams:
            # get the token current value in case there is
            token, val = tok_val.items()[0]
            t_dic = self.found_trigrams.get(token, {})

            # add the values of this lexicon to this token and save it
            t_dic[lex] = val
            self.found_trigrams[token] = t_dic

        # refresh the total trigrams
        self.found_trigrams_total = len(self.found_trigrams.keys())

    def save_non_contiguous_found(self, lex, non_contiguous):
        """
        Save the non_contiguous that were found in each dictionary
        """
        # get the lexicon info if it exists
        tmp = self.found_non_contiguous_by_lex.get(lex, {})
        tmp_non_contiguous = tmp.get('non_contiguous', [])
        tmp_total = tmp.get('total', 0)
    
        # update info
        tmp_non_contiguous += non_contiguous
        tmp_total += len(non_contiguous)

        # save it again
        tmp['non_contiguous'] = tmp_non_contiguous
        tmp['total'] = tmp_total
        self.found_non_contiguous_by_lex[lex] = tmp

        # for each token, save what was the value given by a dictionary
        for tok_val in non_contiguous:
            # get the token current value in case there is
            token, val = tok_val.items()[0]
            t_dic = self.found_non_contiguous.get(token, {})

            # add the values of this lexicon to this token and save it
            t_dic[lex] = val
            self.found_non_contiguous[token] = t_dic

        # refresh the total non_contiguous
        self.found_non_contiguous_total = len(self.found_non_contiguous.keys())

    def save_lex_total_feaures(self, lex, features):
        """
        Save the lexicon features created
        Params:
            lex: lexicon name
            features: final features created by the lexicon
        """
        self.final_lex_features[lex] = features

    def save_lex_sent_match(self, lex, match):
        """
        Save if a lexicon matched the sentiment
        """
        self.lex_sent_match[lex] = match
   
    def get_token_features(self):
        """
        Create a dictionary with all the words marked as true
        """
        features = {}
        for token in self.tokens:
            features[token] = True
            
        return features
    
    def get_stats_features(self):
        """
        Create a dictionary with the statistical features
        """
        features = {}
        features["STS_ALL_CAPS"] = self.all_caps_num
        features["STS_MENTIONS"] = self.mentions_num
        features["STS_HASH"] = self.hash_num
        
        return features

    def get_tokens_lex_features(self):
        """
        Return individual tokens and their values for each of the lexicons
        """
        final = {}

        # check if there exists the tokens processing
        if not self.found_tokens:
            print("You must call save_tokens_found function when processing the lexicons")
            print self.msg_cleaned

        # traverse the tokens/lexicons combination to create individual features
        for token, lexicons in self.found_tokens.iteritems():
            # iterate the dictionary of keypairs lexicon, values
            for lex_name, lex_value in lexicons.iteritems():
                # check if this is a direct value (polarity) or if it has more information
                if type(lex_value) != dict:
                    final[lex_name+"_"+token] = lex_value
                else:
                    # the lexicons have a list of features
                    for sent, val in lex_value.iteritems():
                        final[lex_name+"_"+token+"_"+sent] = val

        return final

    def get_all_features(self):
        """
        Creates a single dictionary with all the features
        """
        features = {}
        features.update(self.get_token_features())
        #features.update(self.get_stats_features())
        #features.update(self.get_tokens_lex_features())
        
        return features
# based on the script of 'KevinZhao'
# You can use this to run the CMU Tweet NLP package(http://www.ark.cs.cmu.edu/TweetNLP/)
# First, download the package at https://github.com/brendano/ark-tweet-nlp/
# Second, put everything in the project directory where you are running the python script

import subprocess
import codecs
import os
import psutil
import tempfile
from libs.files import *
from libs.utils import tprint, Map
import tempfile
import shlex

import sys

# add the tested libs
from inspect import getsourcefile
import os.path as path
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
parent_dir = current_dir[:current_dir.rfind(path.sep, )]

# directories definitions
PROCESSED_DIR = path.join(parent_dir,'2-Processed')
ARK_DIR = path.join(parent_dir,'4-Resources','CMUTaggerPos')
INPUT_DIR = path.join(parent_dir, '1-Input')

# jar location
jar_file = path.join(ARK_DIR,'ark-tweet-nlp-0.3.2.jar')
RUN_TAGGER_CMD = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar "' + jar_file + '" '

#train_files = ['tweeti-b.dist.full.tsv', 'tweeti-b.dev.dist.full.tsv']
#test_files = ['twitter-test-input-B.tsv']
train_files = ['trainingData-B.tsv', 'devData-B.tsv']
test_files = ['testData-B.tsv']

def runFile(train, test):
    # read the file and save only the text
    tweets = []
    tprint("Loading training dataset...")
    for i in train:
        tr = TweetReader(path.join(INPUT_DIR, i), task='B')
        tweets.extend(list(tr))
    
    # convert and save the pos tags
    output = path.join(PROCESSED_DIR, train[0].replace('.csv', '.tagged.tsv').replace('.tsv', '.tagged.tsv'))
    save_pos(tweets, output)
    tprint("done training dataset")
    
    # read test file
    tprint("Loading test dataset...")
    test_tweets = []
    for i in test:
        tr = TweetReader(path.join(INPUT_DIR, i), task='B')
        test_tweets.extend(list(tr))
        
    # convert and save the pos tags
    output = path.join(PROCESSED_DIR, test[0].replace('.tsv', '.tagged.tsv'))
    save_pos(test_tweets, output)
    tprint("done test dataset")
    
def save_pos(tweets, output):
    
    with codecs.open(output,'w','utf-8') as o:
        tprint('Writing output to {}'.format(output))
        for line in get_pos(tweets):
            o.write(line) 

def get_pos(tweets):
    
    tprint("Tag the tweets")
    tweets_text = [ t['text'] for t in tweets]
    tagged = runtagger_parse(tweets_text)
    
    # convert the preivous triple (token, pos, confidence) to 'token\tpos'
    for t, tags in zip(tweets, tagged):
        out = t['sid'] + '\t'
        out += '\t'.join([ ' '.join(tag[:2]) for tag in tags ])
        out += '\n'
        yield out
    

def runFile_():
    # read the file and save only the text
    #tweets = []
    #tprint("Loading training dataset...",)
    #tr = TweetReader('../1-Input/tweeti-b.dist.full.tsv', task='B')
    #tweets.extend(list(tr))
    #tr = TweetReader('../1-Input/tweeti-b.dev.dist.full.tsv', task='B')
    #tweets.extend(list(tr))
    #tprint("done")
    # the jar location
    
    
    # create a processed file for each of the input files
    for f in input_files:
        # create file names
        filename = path.join(parent_dir, INPUT_DIR, f)
        output = path.join(parent_dir, PROCESSED_DIR, f.replace('.tsv', '.conv.tsv'))
        
        # prepare the communication with eternal java .jar
        command = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar "' + jar_file + '" --output-format conll --input-field 4 "' + filename + '"'
        p = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE)
        
        # read the info and save
        #file_name = output
        id_idx = 0
        with codecs.open(output,'w','utf-8') as o:
            tprint('Writing output')
            import pudb
            pudb.set_trace()
            while p.poll() is None:
                l = p.stdout.readline()
                
                if len(l) >0:
                    o.write(l)
                    
                    

#def runString(s):
#    file_name = 'temp_file_%s.txt' % os.getpid()
#    o = codecs.open(file_name,'w','utf-8')
#    uniS = s.decode('utf-8')
#    o.write(uniS)
#    o.close()
#    l = ''
#    p = subprocess.Popen('java -XX:ParallelGCThreads=2 -Xmx500m -jar ' + ARK_DIR +'ark-tweet-nlp-0.3.2.jar/ark-tweet-nlp-0.3.2.jar ' + file_name,stdout=subprocess.PIPE)
#
#    while p.poll() is None:
#        l = p.stdout.readline()
#        break

#    p.kill()
#    psutil.pids()

#    os.remove(file_name)
    #Running one tweet at a time takes much longer time because of restarting the tagger
    #we recommend putting all sentences into one file and then tag the whole file, use the runFile method shown above
#    return l

def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0].decode('utf-8')
                tags = parts[1]
                confidence = float(parts[2].replace(',', '.'))
                yield tokens, tags, confidence


def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""

    # remove carriage returns as they are tweet separators for the stdin
    # interface
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    message = message.encode('utf-8')

    # build a list of args
    args = shlex.split(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # old call - made a direct call to runTagger.sh (not Windows friendly)
    #po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')

    pos_result = result[0].strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    return pos_results


def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results(pos_raw_result)])
    return pos_result


def check_script_is_present(run_tagger_cmd=RUN_TAGGER_CMD):
    """Simple test to make sure we can see the script"""
    success = False
    try:
        args = shlex.split(run_tagger_cmd)
        args.append("--help")
        po = subprocess.Popen(args, stdout=subprocess.PIPE)
        # old call - made a direct call to runTagger.sh (not Windows friendly)
        #po = subprocess.Popen([run_tagger_cmd, '--help'], stdout=subprocess.PIPE)
        while not po.poll():
            lines = [l for l in po.stdout]
        # we expected the first line of --help to look like the following:
        assert "RunTagger [options]" in lines[0]
        success = True
    except OSError as err:
        print "Caught an OSError, have you specified the correct path to runTagger.sh? We are using \"%s\". Exception: %r" % (run_tagger_cmd, repr(err))
    return success    
    
# do the conversion
if __name__ == '__main__':
    # check params
    #if len(sys.argv)!=3:
    #    print 'Execute python CMUPosToken.py infile outfile'
    #    sys.exit()
    
    # get params
    #inp = sys.argv[1]
    #+out = sys.argv[2]
    
    # make the conversion for train, test
    runFile(train_files, test_files)
    
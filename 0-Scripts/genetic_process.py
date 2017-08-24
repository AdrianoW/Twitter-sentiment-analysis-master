import random
from deap import base, creator, tools, algorithms
import libs.resources as r
import numpy as np
from libs.utils import tprint
import libs.files as fh
import scipy as scy

# Simple definition
# available_lex = [ r for r in r.lexs if r.opinion]
available_lex = [lex for lex in r.lexs]
IND_SIZE = len(available_lex) + 1 # liwc is not inside the lib
POP_SIZE = 30
MU = int(POP_SIZE/5)
LAMBDA_= POP_SIZE-MU
rnd_seed = 9000
LAST_POPULATION =  '../3-Output/population_new1.csv'
POPULATION = '../3-Output/population.csv'
NGEN = 30

def rand():
    """
    My own random that enables the gene 10% of the time
    """
    return int(random.randint(0,10) == 10)

def print_evolution(log):
    """
    Print the evolution of the populations
    """
    import matplotlib.pyplot as plt
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()

def load_lex_features_dump(lex):
    """
    Load the saved trained and test lexicon features
    """
    tmp_train = pipe.load_dump_data(lex.prefix+'_train.pck')
    tmp_dev = pipe.load_dump_data(lex.prefix+'_dev.pck')
    tmp_test = pipe.load_dump_data(lex.prefix+'_test.pck')
    
    return tmp_train, tmp_dev, tmp_test

def read_original_files():
    """
    Read the original files 
    """
    # read the config file
    cfg = fh.read_config_file("all.yaml")

    # read the train file
    file_name = "../1-Input/trainingData-B.tsv"
    file_type = 'B'
    train_feat, labels, train_tweets = pipe.create_features(file_name, file_type, cfg)

    # read the dev file
    file_name = "../1-Input/devData-B.tsv"
    file_type = 'B'
    dev_feat, dev_labels, dev_tweets = pipe.create_features(file_name, file_type, cfg)

    # read the test file
    file_name = "../1-Input/testData-B.tsv"
    file_type = 'B'
    test_feat, gold, test_tweets = pipe.create_features(file_name, file_type, cfg)

    return train_feat, dev_feat, test_feat

def load_LIWC():
    """
    Load the LIWC from HD 
    """
    zero = {'ABBREVIATIONS': 0.0, 'ACHIEVEMENT': 0.0, 'AFFECTIVE_PROCESS': 0.0, 'ALLPCT': 0.0, 'ANGER': 0.0, 'ANXIETY': 0.0, 'APOSTRO': 0.0, 'ARTICLES': 0.0, 'ASSENTS': 0.0, 'BODY_STATES': 0.0, 'CAUSATION': 0.0, 'CERTAINTY': 0.0, 'COGNITIVE_PROCESS': 0.0, 'COLON': 0.0, 'COMMA': 0.0, 'COMMUNICATION': 0.0, 'DASH': 0.0, 'DEATH_AND_DYING': 0.0, 'DIC': 0.0, 'DISCREPANCY': 0.0, 'DOWN': 0.0, 'EATING': 0.0, 'EMOTICONS': 0.0, 'EXCLAM': 0.0, 'EXCLUSIVE': 0.0, 'FAMILY': 0.0, 'FEELING': 0.0, 'FILLERS': 0.0, 'FRIENDS': 0.0, 'FUTURE': 0.0, 'GROOMING': 0.0, 'HEARING': 0.0, 'HOME': 0.0, 'HUMANS': 0.0, 'I': 0.0, 'INCLUSIVE': 0.0, 'INHIBITION': 0.0, 'INSIGHT': 0.0, 'JOB_OR_WORK': 0.0, 'LEISURE_ACTIVITY': 0.0, 'METAPHYSICAL': 0.0, 'MONEY': 0.0, 'MOTION': 0.0, 'MUSIC': 0.0, 'NEGATIONS': 0.0, 'NEGATIVE_EMOTION': 0.0, 'NONFLUENCIES': 0.0, 'NUMBERS': 0.0, 'OCCUPATION': 0.0, 'OPTIMISM': 0.0, 'OTHER': 0.0, 'OTHERP': 0.0, 'PARENTH': 0.0, 'PAST': 0.0, 'PERIOD': 0.0, 'PHYSICAL_STATES': 0.0, 'POSITIVE_EMOTION': 0.0, 'POSITIVE_FEELING': 0.0, 'PREPOSITIONS': 0.0, 'PRESENT': 0.0, 'PRONOUN': 0.0, 'QMARK': 0.0, 'QMARKS': 0.0, 'QUOTE': 0.0, 'REFERENCE_PEOPLE': 0.0, 'RELIGION': 0.0, 'SADNESS': 0.0, 'SCHOOL': 0.0, 'SEEING': 0.0, 'SELF': 0.0, 'SEMIC': 0.0, 'SENSORY_PROCESS': 0.0, 'SEXUALITY': 0.0, 'SIXLTR': 0.0, 'SLEEPING': 0.0, 'SOCIAL_PROCESS': 0.0, 'SPACE': 0.0, 'SPORTS': 0.0, 'SWEAR_WORDS': 0.0, 'TENTATIVE': 0.0, 'TIME': 0.0, 'TV_OR_MOVIE': 0.0, 'UNIQUE': 0.0, 'UP': 0.0, 'WC': 0.0, 'WE': 0.0, 'WPS': 0.0, 'YOU': 0.0}

    train_liwc = fh.read_liwc('../2-Processed/allTrainingData.liwc.tsv')
    train_liwc1 = fh.read_liwc('../2-Processed/trainingData-B.liwc.tsv')
    dev_liwc = fh.read_liwc('../2-Processed/devData-B.liwc.tsv')
    test_liwc = fh.read_liwc('../2-Processed/testData-B.liwc.tsv')

    # as all the information comes out of orde from liwc, it needs to be put in order again.
    train_feat, dev_feat, test_feat = read_original_files()
    train_sent_liwc = [ train_liwc.get(int(tw.sid), train_liwc1.get(int(tw.sid), zero)) for tw in train_feat]
    dev_sent_liwc = [ dev_liwc[int(tw.sid)] for tw in dev_feat]
    test_sent_liwc = [ test_liwc[int(tw.sid)] for tw in test_feat]

    return train_sent_liwc, dev_sent_liwc, test_sent_liwc

def create_ind_features(ind, base_train, base_dev, base_test):
    """
    Given an individual, create the features according to its genes
    Params:
        ind: individual generated in the genetic algorithm
    Returns:
        train and test data as a scipy sparse matrix
    """
    # remove the last locus as it refers to the LIWC
    LIWC_gene = ind[-1]
    tmp_ind = ind[:-1]

    # load the lexicons according to the gens
    ind_lexs = [available_lex[idx] for idx, present in enumerate(tmp_ind) if present]
    
    # for each of the lexicons, merge them
    final_train = base_train.copy()
    final_dev = base_dev.copy()
    final_test = base_test.copy()

    for lex in ind_lexs:
        # try:
        lex_train, lex_dev, lex_test = load_lex_features_dump(lex)
        final_train, final_dev = pipe.join_lex_features(final_train, lex_train, final_dev, 
                                                    lex_dev, verbose=False, create_vec=False)
        _, final_test = pipe.join_lex_features(final_train, lex_train, final_test, 
                                                    lex_test, verbose=False, create_vec=False)
        # except:
        #     # import pudb; pudb.set_trace()
        #     print 'error loading ind', ind
        #     print lex

    # check if should add LIWC
    if LIWC_gene:
        final_train, final_dev = pipe.join_lex_features(final_train, train_liwc, 
                                                        final_dev, dev_liwc, 
                                                        verbose=False, create_vec=True)
        _, final_test = pipe.join_lex_features(final_train, train_liwc, 
                                               final_test, test_liwc, 
                                               verbose=False, create_vec=True)


    return final_train, final_dev, final_test

# define an individual gene
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# initialize the simple evolution toolbox
simple_tb = base.Toolbox()

# define the individual generator
simple_tb.register("attr_bool", rand)
simple_tb.register("individual", tools.initRepeat, creator.Individual, 
                  simple_tb.attr_bool, n=IND_SIZE)

# define the population generator
simple_tb.register("population", tools.initRepeat, list, simple_tb.individual)
pop = simple_tb.population(POP_SIZE)

# change the last individual to be all genes set
for i in range(IND_SIZE):
    pop[-1][i] = 1
print pop[-1]

# create the evolutionary parts
simple_tb.register("mate", tools.cxOnePoint) # mate changing in one point
simple_tb.register("mutate", tools.mutFlipBit, indpb=0.20) # flip a bit with 10% chance
simple_tb.register("select", tools.selBest)

# register the statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
#stats.register("std", numpy.std)
stats.register("min", np.min)
stats.register("max", np.max)

# save the best
hof = tools.HallOfFame(5)

###
# Lexicon Selection
##

import libs.pipeline as pipe
import libs.resources as r
import libs.utils as u

# read the base data
train_dataset_stop = pipe.load_dump_data('train_base_data.pck')
dev_dataset_stop = pipe.load_dump_data('dev_base_data.pck')
test_dataset_stop = pipe.load_dump_data('test_base_data.pck')

labels = pipe.load_dump_data('labels.pck')
dev_labels = pipe.load_dump_data('dev_labels.pck')
gold = pipe.load_dump_data('gold.pck')

# read LIWC features
train_liwc, dev_liwc, test_liwc = load_LIWC()

import sklearn.model_selection as model_selection

# machine learning related part
skf = model_selection.StratifiedKFold(10, random_state=rnd_seed)
base_train = train_dataset_stop
base_dev = dev_dataset_stop
base_test = test_dataset_stop

import os.path as path
if path.exists(LAST_POPULATION):
    tprint('Recovering individual values from file %s' % LAST_POPULATION)
    all_individuals = {}
    with open(LAST_POPULATION) as f:
        for line in f:
            ind, val = line.strip().split(';')
            all_individuals[ind] = eval(val)
else:
    tprint('Starting with no previous started value')
    all_individuals = {}

def fitness_function(individual):#, base_train, base_test, labels, gold):
    """
    Given an individual, make the fitness test and find a value that defines this individual
    """    
    # check if this individual was tested before
    key = '{}'.format(individual)
    prev = all_individuals.get(key, None)
    if prev:
        (final_val, dev_val, idx_max) = prev
        tprint('Calculated already \t {} \t {}'.format(key, (final_val, dev_val, idx_max)))#, (train_val, test_val)))
        return dev_val,
    else:
        # create the feature dataset
        train_data, dev_data, test_data = create_ind_features(individual, base_train, base_dev, base_test)
        # _, test_data = create_ind_features(individual, base_train, base_test)

        # run the experiment with multiple algorithms
        ret_df = pipe.run_multiple_class(train_data, labels, dev_data, dev_labels, rnd_seed=rnd_seed)

        # get the best available value
        idx_max = ret_df['dev score'].argmax()
        final_val = ret_df.loc[idx_max, 'train score']
        dev_val =  ret_df.loc[idx_max, 'dev score']

        # run the experiment with the test dataset
        # X_train = scy.sparse.csr_matrix(scy.sparse.vstack((train_data, dev_data)))
        # y_train = np.concatenate((labels, dev_labels))

        # run the experiment with multiple algorithms
        # ret_df = pipe.predict_test_multi_proc(idx_max, X_train, y_train, test_data, gold, rnd_seed=rnd_seed, test_name='test')

        all_individuals[key] = ( (final_val, dev_val, idx_max) ) #, (ret_df.loc[idx_max, 'train score'], ret_df.loc[idx_max, 'test score']) )
        tprint('New Individual \t {} \t {}'.format(key, (final_val, dev_val, idx_max))) #,  (ret_df.loc[idx_max, 'train score'], ret_df.loc[idx_max, 'test score'])))
        return dev_val,

# register the evaluation function and run the algorithm
simple_tb.register("evaluate", fitness_function)
final_pop, log = algorithms.eaMuPlusLambda(pop, simple_tb, 
                                            mu=MU, lambda_=LAMBDA_,
                                            cxpb=0.5, mutpb=0.2, 
                                            ngen=NGEN,
                                            stats=stats, halloffame=hof, 
                                            verbose=True)
# final_pop, log = algorithms.eaMuCommaLambda(pop, simple_tb, 
#                                             mu=MU, lambda_=LAMBDA_,
#                                             cxpb=0.5, mutpb=0.2, 
#                                             ngen=NGEN,
#                                             stats=stats, halloffame=hof, 
#                                             verbose=True)

# save the population
tprint('Saving population to file %s' % POPULATION)
with open(POPULATION, 'w') as f:
    results = sorted(all_individuals.iteritems(), key=lambda (k,v): k[1], 
                    reverse=True)
    for line in results:
        k,v = line
        f.write('{};{}\n'.format(k,v))

tprint('Saving log to file %s.log' % POPULATION)
with open(POPULATION+'.log', 'w') as f:
    f.write("gen; avg; min; max\n")
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    for g, a, mi, mx in zip(gen, avg, min_, max_ ):
        f.write('{};{};{};{}\n'.format(g, a, mi, mx ))


#print the evolution log
# print_evolution(log)

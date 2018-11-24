from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
from load_map import *
import matplotlib.pyplot as plt
import numpy as np
from math import log;
from pylab import mean;
from scipy.stats import spearmanr

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def dictToVector(d):
    vector = []
    for k, v in sorted(d.items()):
        vector.append(v)
    return vector

def PMI(c_xy, c_x, c_y, N):
  '''Compute the pointwise mutual information using cooccurrence counts.

  :type c_xy: int 
  :type c_x: int 
  :type c_y: int 
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value
  
  '''
  return log((N * c_xy) / (c_x * c_y), 2)

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")
    
def equalizeDictionaries(d0, d1):
    tmp = {}
    for wid, pmi in d0.items():
        if(wid not in d1):
            tmp.update({wid: pmi})
    for key, value in tmp.items():
        d0.pop(key)
    tmp = {}
    for wid, pmi in d1.items():
        if(wid not in d0):
            tmp.update({wid: pmi})
    for key, value in tmp.items():
        d1.pop(key)
    return d0, d1
    
def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.  
  v0, v1 = equalizeDictionaries(v0, v1)

  v0 = dictToVector(v0)
  v1 = dictToVector(v1)
    
  return np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

def diceMeasure(v0, v1): 
    intersection = set()
    
    for wid0, count0 in v0.items():
        for wid1, count1 in v1.items():
            if(wid0 == wid1):
                intersection.add(wid0)
                break
    
    return (2 * len(intersection)) / (len(v0) + len(v1))

def jaccardMeasure(v0, v1):        
    intersection = set()
    union = set()
    
    for wid0, count0 in v0.items():
        for wid1, count1 in v1.items():
            if(wid0 == wid1):
                union.add(wid0)
                intersection.add(wid0)
                break
            else:
                 union.add(wid0)
                 union.add(wid1)

    return len(intersection) / len(union)

def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id

    '''
    vectors = {}
    for wid0 in wids:
        temp = {}
        for wid1 in co_counts[wid0]:
            #if(wid0 != wid1):
            temp.update({wid1:co_counts[wid0][wid1] * (PMI(co_counts[wid0][wid1], o_counts[wid0], o_counts[wid1], tot_count))})
        vectors[wid0] = temp
    return vectors

def getWordCounts(wids, counts, co_counts, tot_count):
    vectors = {}
    for wid0 in wids:
        temp = {}
        for wid1 in wids:
            if(wid0 != wid1):
                temp.update({wid1:co_counts[wid0][wid1]})
        vectors[wid0] = temp
    return vectors

def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.

  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''  
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
                                         o_counts[pair[0]],o_counts[pair[1]]))

def freq_v_sim(sims):
  xs = []
  ys = []
  for pair in sims.items():
    ys.append(pair[1])
    c0 = o_counts[pair[0][0]]
    c1 = o_counts[pair[0][1]]
    xs.append(min(c0,c1))
  plt.clf() # clear previous plots (if any)
  plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
  plt.plot(xs, ys, 'k.') # create the scatter plot
  plt.xlabel('Min Freq')
  plt.ylabel('Similarity')
  print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
  print("P-Value = {:.2f}".format(spearmanr(xs,ys)[1]))
  plt.show() #display the set of plots

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]

def newPairs(words):
    pairs = []
    for word0, word1 in words.items():
        pairs.append((word2wid[tw_stemmer(word0)], word2wid[tw_stemmer(word1)]))
    return pairs

def makeWidSet(wordDict):
    widSet = set()
    for w0, w1 in wordDict.items():
        widSet.add(word2wid[tw_stemmer(w0)])
        widSet.add(word2wid[tw_stemmer(w1)])
    return widSet

test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]

similar_words = {'poodle': 'shepherd', 'gala': 'party', 'puppy': 'pet', 'siren': 'mermaid', 'wheat': 'seed', 'occasion': 'reunion', 'fur': 'fox', 'clay': 'wood', 'vegetable': 'tomato', 'autumn': 'spring', 'peach': 'berry', 'wizard': 'magic', 'trail': 'heel', 'tail': 'cat', 'lemon': 'lime', 'lunch': 'dinner', 'pie': 'cake', 'doll': 'toy', 'birth': 'birthday', 'purple': 'blue', 'christmas': 'holiday', 'festival': 'celebration', 'gift': 'present', 'shoe':'hat'}
dissimilar_words = {'botox': 'volcano', 'pear': 'safari', 'instinct':'miner', 'priest':'passport', 'panther':'wage', 'vegan':'sword', 'corruption':'mouse', 'paranormal':'council', 'professor':'farm', 'election':'coffee', 'boat':'candy', 'lack':'garden', 'speech': 'rocket', 'sea':'intern', 'protest':'pour', 'june':'peace', 'drama':'milk', 'gym':'lol', 'score':'home', 'college':'shrimp', 'honey':'airport', 'violin':'cow', 'prison':'star'}
moderate_words = {'good':'keyboard', 'happy':'job', 'spoon':'cabinet', 'memory':'dread', 'dream':'deep',
                  'phone':'chair', 'door':'dog', 'eartquake':'ocean', 'spider':'crime', 'tsunami':'president', 'obama':'email',
                  'time':'york', 'cereal':'newspaper', 'perfume':'child', 'paparazzi':'court', 'house':'boat', 'roll':'die', 'florida':'knight', 'greece':'flood', 'pasta':'scotland', 'google':'reader', 'twitter':'politics','siri':'friend'}
allSimilar = makeWidSet(similar_words)
allDissimilar = makeWidSet(dissimilar_words)
allModerate = makeWidSet(moderate_words)    

stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)

similarPairs = newPairs(similar_words)
dissimilarPairs = newPairs(dissimilar_words)
moderatePairs = newPairs(moderate_words)    

#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", set(list(all_wids) + list(allSimilar) + list(allModerate) + list(allDissimilar)))

#make the word vectors
ppmis = create_ppmi_vectors(all_wids, o_counts, co_counts, N)

similarPPMIS = create_ppmi_vectors(allSimilar, o_counts, co_counts, N)
dissimilarPPMIS = create_ppmi_vectors(allDissimilar, o_counts, co_counts, N)
moderatePPMIS = create_ppmi_vectors(allModerate, o_counts, co_counts, N)

def findAllSimilarities(o_counts, co_counts, ppmis, pairs, similarity):
    # compute cosine similarites for all pairs we consider
    c_sims = {(wid0,wid1): cos_sim(ppmis[wid0],ppmis[wid1]) for (wid0,wid1) in pairs}
    dice_sims = {(wid0,wid1): diceMeasure(co_counts[wid0],co_counts[wid1]) for (wid0,wid1) in pairs}
    jaccard_sims = {(wid0,wid1): jaccardMeasure(co_counts[wid0], co_counts[wid1]) for (wid0,wid1) in pairs}
    
    print("============================" + similarity + "===========================")
    print("Sort by cosine similarity")
    print_sorted_pairs(c_sims, o_counts)
    freq_v_sim(c_sims)
    
    print("\nSort by dice similarity")
    print_sorted_pairs(dice_sims, o_counts)
    freq_v_sim(dice_sims)
    
    print("\nSort by Jaccard similarity")
    print_sorted_pairs(jaccard_sims, o_counts)
    freq_v_sim(jaccard_sims)
    

findAllSimilarities(o_counts, co_counts, ppmis, wid_pairs, 'OG TEST WORDS')
findAllSimilarities(o_counts, co_counts, similarPPMIS, similarPairs, 'SIMILAR WORDS')
findAllSimilarities(o_counts, co_counts, dissimilarPPMIS, dissimilarPairs, 'DISSIMILAR WORDS')
findAllSimilarities(o_counts, co_counts, moderatePPMIS, moderatePairs, 'MODERATELY SIMILAR WORDS')

############################################################
def dictConvert(dictionary):
    result = dict((v,k) for k,v in dictionary.items())
    return [result[key] for key in sorted(result)]        

def getWordRank(word):
    word = tw_stemmer(word)
    sortedCounts = reversed(dictConvert(o_counts))
    index = 0
    for i in sortedCounts:
        if(i == word2wid[word]):
            print("Word is ranked at ", index)
            break
        index += 1

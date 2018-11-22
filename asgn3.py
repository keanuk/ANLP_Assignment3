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

  #You will need to replace with the real function
  v0 = dictToVector(v0)
  v1 = dictToVector(v1)
  return np.dot(v0, v1) / (np.linalg.norm(v1) * np.linalg.norm(v0))

def cos_sim_list(v0,v1):
  return np.dot(v0, v1) / (np.linalg.norm(v1) * np.linalg.norm(v0))

def diceMeasure(v0, v1):
    v0 = dictToVector(v0)
    v1 = dictToVector(v1)
    return (2 * len(np.intersect1d(v0, v1))) / (len(v0) + len(v1))

def jaccardMeasure(v0, v1):
    v0 = dictToVector(v0)
    v1 = dictToVector(v1)
    return len(np.intersect1d(v0, v1)) / len(np.union1d(v0, v1))

def buildVectors(word0, word1):
    d0 = co_counts[word2wid[word0]]
    d1 = co_counts[word2wid[word1]]
    
    counts0 = {}
    counts1 = {}
    pmi0 = {}
    pmi1 = {}
    
    index = 0
    for k0, v0 in sorted(d0.items()):
        for k1, v1 in sorted(d1.items()):
            if k0 == k1:
                #print("Found a key match: ", k0, k1, " with values: ", v0, v1)
                #print("Checking PMI with c_xy = ", k0)
                #print("Checking PMI with c_x = ", o_counts[word2wid[word0]])
                #print("Checking PMI with c_y = ", o_counts[k0])
                #print("Checking PMI with N = ", N)
                counts0.update({index:v0})
                counts1.update({index:v1})
                pmi0.update({index:PMI(d0[k0], o_counts[word2wid[word0]], o_counts[k0], N)})
                pmi1.update({index:PMI(d1[k1], o_counts[word2wid[word1]], o_counts[k1], N)})
                index += 1
                break
    return counts0, counts1, pmi0, pmi1


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
        for wid1 in wids:
            if(wid0 != wid1):
                vectors[wid0] = {wid1:(PMI(co_counts[wid0][wid1], o_counts[wid0], o_counts[wid1], tot_count))}
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
#  plt.show() #display the set of plots

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''
  return [(x, y) for x in items for y in items if x < y]


test_words = ["cat", "dog", "mouse", "computer","@justinbieber", "red", "blue"]
stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)


#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

#make the word vectors
vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)

# compute cosine similarites for all pairs we consider
c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

print("Sort by cosine similarity")
print_sorted_pairs(c_sims, o_counts)

def countPMI(count, pmi):
    vector = []
    index = 0
    for key, value in count.items():
        vector.append(pmi[index] * value)
        index += 1
    return vector

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

def wordCompare():
    c0, c1, pmi0, pmi1 = buildVectors("red", "dog")

    #print("\n\n\nCounts 0:\n\n\n ", c0)
    #print("\n\n\nCounts 1:\n\n\n ", c1)
    #print("\n\n\nPMI 0:\n\n\n ", pmi0)
    #print("\n\n\nPMI 1:\n\n\n ", pmi1)

    cpmi0 = countPMI(c0, pmi0)
    cpmi1 = countPMI(c1, pmi1)

    print("Cosine similarity: ", cos_sim_list(cpmi0, cpmi1))
    print("Dice measure: ", diceMeasure(c0, c1))
    print("Jaccard measure: ", jaccardMeasure(c0, c1))

wordCompare()        

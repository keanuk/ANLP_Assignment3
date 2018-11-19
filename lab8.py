'''
Authors: Luke Shrimpton, Sharon Goldwater, Ida Szubert
Date: 2014-11-01, 2017-11-05
Copyright: This work is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License
(http://creativecommons.org/licenses/by-nc/4.0/): You may re-use,
redistribute, or modify this work for non-commercial purposes provided
you retain attribution to any previous author(s).
'''
from __future__ import division;
from math import log;
from math import sqrt;
from pylab import mean;
from load_map import *
import numpy as np
from sklearn.metrics import jaccard_similarity_score

def PMI(c_xy, c_x, c_y, N):
    # Computes PMI(x, y) where
    # c_xy is the number of times x co-occurs with y
    # c_x is the number of times x occurs.
    # c_y is the number of times y occurs.
    # N is the number of observations.
    
    #print("Calculating PMI with c_xy: ", c_xy)
    #print("Calculating PMI with c_x: ", c_x)
    #print("Calculating PMI with c_y: ", c_y)
    #print("Calculating PMI with N: ", N)

    return log((N * c_xy) / (c_x * c_y), 2)

def dictToVector(d):
    vector = []
    for k, v in sorted(d.items()):
        vector.append(v)
    return vector
        

def cos_sim(v0, v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  
  '''
  v0 = dictToVector(v0)
  v1 = dictToVector(v1)
  return np.dot(v0, v1) / (np.linalg.norm(v1) * np.linalg.norm(v0))

def diceMeasure(v0, v1):
    v0 = dictToVector(v0)
    v1 = dictToVector(v1)
    return (2 * len(np.intersect1d(v0, v1))) / (len(v0) + len(v1))

def jaccardMeasure(v0, v1):
    v0 = dictToVector(v0)
    v1 = dictToVector(v1)
    print("Jaccard intersection: ", np.intersect1d(v0, v1))
    return len(np.intersect1d(v0, v1)) / len(np.union1d(v0, v1))

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

# List of positive words:
pos_words = ["love"];
# List of negative words:
neg_words = ["hate"];
# List of target words:
targets = ["@justinbieber", "husband", "wife", "#1000aday", "red", "blue", "phone", "mop", "satan"];

# Collect all words of interest and store their term ids:
all_words = set(pos_words+neg_words+targets);
all_wids = set([word2wid[x] for x in all_words]);

# Define the data structures used to store the counts:
o_counts = {}; # Occurrence counts
co_counts = {}; # Co-occurrence counts

# Load the data:
fp = open("/afs/inf.ed.ac.uk/group/teaching/anlp/lab8/counts", "r");
lines = fp.readlines();
N = float(lines[0]); # First line contains the number of observations.
for line in lines[1:]:
    line = line.strip().split("\t");
    wid0 = int(line[0]);
    o_counts[wid0] = int(line[1])
    if(wid0 in all_wids): # Only get/store counts for words we are interested in
        #o_counts[wid0] = int(line[1]); # Store occurence counts
        co_counts[wid0] = dict([[int(y) for y in x.split(" ")] for x in line[2:]]); # Store co-occurence counts
        


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

def wordCompare():
    c0, c1, pmi0, pmi1 = buildVectors("mop", "satan")

    print("\n\n\nCounts 0:\n\n\n ", c0)
    print("\n\n\nCounts 1:\n\n\n ", c1)
    print("\n\n\nPMI 0:\n\n\n ", pmi0)
    print("\n\n\nPMI 1:\n\n\n ", pmi1)

    
    print("Cosine similarity: ", cos_sim(pmi0, pmi1))
    print("Dice measure: ", diceMeasure(c0, c1))
    print("Jaccard measure: ", jaccardMeasure(c0, c1))

        
lab8()
print("thank u, next")
wordCompare()

def lab8():
    for target in targets:
        targetid = word2wid[target]
        posPMIs = []
        negPMIs = []
        # compute PMI between target and each positive word, and
        # add it to the list of positive PMI values
        for pos in pos_words:
            tcoDict = co_counts[targetid]
            tco = tcoDict[word2wid[pos]]
            posPMIs.append(PMI(tco, o_counts[targetid], o_counts[word2wid[pos]], N))
        # same for negative words
        for neg in neg_words:
            tcoDict = co_counts[targetid]
            tco = tcoDict[word2wid[neg]]
            negPMIs.append(PMI(tco, o_counts[targetid], o_counts[word2wid[neg]], N))
    #uncomment the following line when posPMIs and negPMIs are no longer empty.
        print(target, ": ", mean(posPMIs), "(pos), ", mean(negPMIs), "(neg)")
        

"""

Testing idea of alpha and distribution

"""

import os, time, json, sys, pdb
import numpy as np

import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets

import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

import mlep.data_encoder.w2vGoogleNews as w2vGoogleNews
import mlep.trackers.MemoryTracker as MemoryTracker

import mlep.text.DataCharacteristics.OnlineSimilarityDistribution as OnlineSimilarityDistribution
import mlep.text.DataCharacteristics.CosineSimilarityDataCharacteristics as CosineSimilarityDataCharacteristics
import mlep.text.DataCharacteristics.L2NormDataCharacteristics as L2NormDataCharacteristics

import warnings
# warnings.filterwarnings(action="ignore", category=FutureWarning)

import collections


_encoder = w2vGoogleNews.w2vGoogleNews()
_encoder.setup()




# we are not updating internal timer...
trainingData = BatchedLocal.BatchedLocal(data_source="./data/initialTrainingData.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
trainingData.load()

X_train = _encoder.batchEncode(trainingData.getData())
X_centroid = _encoder.getCentroid(X_train)

nBins = 40
alpha = 0.6

# Set up initial Distribution
#charac = CosineSimilarityDataCharacteristics.CosineSimilarityDataCharacteristics()
charac = L2NormDataCharacteristics.L2NormDataCharacteristics()
charac.buildDistribution(X_centroid,X_train)


driftWindowTracker = MemoryTracker.MemoryTracker()
driftWindowTracker.addNewMemory(memory_name="gen-mem",memory_store='memory')
driftWindowTracker.addNewMemory(memory_name="core-mem",memory_store='memory')
driftWindowTracker.addNewMemory(memory_name="edge-mem",memory_store='memory')



#streamData = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
streamData = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)

edge_centroid = np.zeros(X_centroid.shape[0])
edge_sum = np.zeros(X_centroid.shape[0])
edge_seen = False

processLength = 0
zones = []


while streamData.next():
    processLength += 1
    
    # For each data, we are checking where it falls on the distribution
    _encoded = _encoder.encode(streamData.getData())
    _distance = _encoder.getDistance(_encoded, X_centroid)

    # Check where it falls:
    if _distance >= charac.delta_low and _distance <= charac.delta_high:
        # Within concentrated zone
        #add to edge memory  and core memory
        driftWindowTracker.addToMemory("edge-mem", streamData.getObject())
        driftWindowTracker.addToMemory("core-mem", streamData.getObject())

        # track centroids of edge-mem to see if it escapes boundary...
        # rolling centroid...
        if edge_seen:
            edge_centroid = edge_centroid + (edge_sum + _encoded)/float(driftWindowTracker.memorySize("edge-mem")) - (edge_sum/(float(driftWindowTracker.memorySize("edge-mem"))-1))
        else:
            #Zero division error/Nans
            edge_centroid = edge_centroid + (edge_sum + _encoded)/float(driftWindowTracker.memorySize("edge-mem"))
            edge_seen = True
        edge_sum += _encoded

        
        
        pass
    elif _distance < charac.delta_low:
        # Inside core data
        # add to core memory 
        driftWindowTracker.addToMemory("core-mem", streamData.getObject())
        pass
    elif _distance > charac.delta_high:
        # outside concentrated zone
        # add to gen memory and edge memory
        driftWindowTracker.addToMemory("gen-mem", streamData.getObject())
        driftWindowTracker.addToMemory("edge-mem", streamData.getObject())
        pass
    

    if edge_seen:
        #We have a valid centroid value for edge-memory
        edge_distance = _encoder.getDistance(edge_centroid, X_centroid)
        if edge_distance >= charac.delta_low and edge_distance <= charac.delta_high:
            #print("In Edge Zone")
            zones.append(1)
        elif edge_distance < charac.delta_low:
            #print("In Core Zone")
            zones.append(0)
        elif edge_distance > charac.delta_high:
            #print("In Gen  Zone")
            zones.append(2)

    

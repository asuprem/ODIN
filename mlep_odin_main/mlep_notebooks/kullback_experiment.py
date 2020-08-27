import os, time, json, sys, pdb

import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets

import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

import mlep.data_encoder.w2vGoogleNews as w2vGoogleNews
import mlep.trackers.MemoryTracker as MemoryTracker

import mlep.drift_detector.UnlabeledDriftDetector.KullbackLeibler as KullbackLeibler
import mlep.text.DataCharacteristics.OnlineSimilarityDistribution as OnlineSimilarityDistribution

import warnings
# warnings.filterwarnings(action="ignore", category=FutureWarning)

import traceback
import collections


def main():
    # update as per experiment requires
    # Checking Kullback Leibler...
    _encoder = w2vGoogleNews.w2vGoogleNews()
    _encoder.setup()


    # we are not updating internal timer...
    #streamData = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    
    trainingData = BatchedLocal.BatchedLocal(data_source="./data/initialTrainingData.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    trainingData.load()

    X_train = _encoder.batchEncode(trainingData.getData())
    X_centroid = _encoder.getCentroid(X_train)

    nBins = 40
    # Set up initial Distribution
    initialDistribution = OnlineSimilarityDistribution.OnlineSimilarityDistribution(nBins)
    for _data in trainingData.getData():
        initialDistribution.update(_encoder.getDistance(_encoder.encode(_data), X_centroid))
    kullback = KullbackLeibler.KullbackLeibler(initialDistribution)

    driftWindowTracker = MemoryTracker.MemoryTracker()
    driftWindowTracker.addNewMemory(memory_name="kullback",memory_store='memory')

    """
    totalCounter = 0
    implicit_mistakes = 0.0
    implicit_count = 0
    explicit_mistakes = 0.0
    explicit_count = 0
    """
    raw_vals=[0]
    dqlen=100.0
    windowed_raw = collections.deque([], int(dqlen))


    #streamData = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    streamData = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    secondary_distribution = OnlineSimilarityDistribution.OnlineSimilarityDistribution(nBins)
    processLength = 0
    #genCount = 0
    #axv = []
    while streamData.next():
        processLength += 1
        # add to memory
        #driftWindowTracker.addToMemory("kullbak", streamData.getObject())

        # Perform drift detection (but just tracking f. now)
        _encoded = _encoder.encode(streamData.getData())
        _distance = _encoder.getDistance(_encoded, X_centroid)
        secondary_distribution.update(_distance)
        raw_val = kullback.detect(_distance, secondary_distribution)
        windowed_raw.append(raw_val)
        #if streamData.streamLength()>3000:
        #    pdb.set_trace()
        #raw_vals.append((raw_vals[-1]+raw_val)/streamData.streamLength())
        raw_vals.append(raw_val)
        #raw_vals.append(raw_val)
        
        """
        driftWindowTracker.addToMemory(memory_name="kullback", data=streamData.getObject())
        if processLength>dqlen:
            if raw_vals[-1] > .02:
                genCount += 1

                print("processed ",streamData.streamLength(), " and detected drift:  ", str(raw_vals[-1]))
                # transfer memory, etc etc
                trainingData = driftWindowTracker.transferMemory("kullback")
                driftWindowTracker.clearMemory("kullback")
                X_train = _encoder.batchEncode(trainingData.getData())
                X_centroid = _encoder.getCentroid(X_train)
                # update distribution
                kullback.reset()
                secondary_distribution = OnlineSimilarityDistribution.OnlineSimilarityDistribution(nBins)
                processLength = 0
        """
            
        """
        driftWindowTracker.addToMemory(memory_name="kullback", data=streamData.getObject())
        if driftWindowTracker.memorySize("kullback") > 3000:
            #print("processed ",streamData.streamLength())
            # transfer memory, etc etc
            trainingData = driftWindowTracker.transferMemory("kullback")
            driftWindowTracker.clearMemory("kullback")
            X_train = _encoder.batchEncode(trainingData.getData())
            X_centroid = _encoder.getCentroid(X_train)
            # update distribution
            kullback.reset()
            secondary_distribution = OnlineSimilarityDistribution.OnlineSimilarityDistribution(nBins)
        """ 
    pdb.set_trace()
        

    
if __name__ == "__main__":
    main()
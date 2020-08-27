import os, time, json, sys, pdb, click

# import mlflow

import mlep.core.MLEPDriftAdaptor as MLEPDriftAdaptor

import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets
import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

import mlep.learning_model.kerasSimple as kerasSimple
import mlep.learning_model.kerasComplex as kerasComplex

import mlep.data_encoder.w2vGoogleNews as w2vGoogleNews


def main():

    io_utils.std_flush("Initialized at %s"%time_utils.readable_time("%H:%M:%S"))
    _encoder = w2vGoogleNews.w2vGoogleNews()
    _encoder.setup()
    io_utils.std_flush("Set up encoder at %s"%time_utils.readable_time("%H:%M:%S"))
    
    trainingData = BatchedLocal.BatchedLocal(data_source='./data/pure_new_dataset.json', data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    trainingData.load()
    io_utils.std_flush("Loaded training data at %s"%time_utils.readable_time("%H:%M:%S"))

    X_train = _encoder.batchEncode(trainingData.getData())
    y_train = trainingData.getLabels()
    io_utils.std_flush("Batch encoded data at %s"%time_utils.readable_time("%H:%M:%S"))

    model = kerasComplex.kerasComplex()
    io_utils.std_flush("Generated model at %s"%time_utils.readable_time("%H:%M:%S"))

    io_utils.std_flush("Starting training at %s"%time_utils.readable_time("%H:%M:%S"))
    precision, recall, score = model.fit_and_test(X_train, y_train)
    io_utils.std_flush("Completed training with precision: %f\trecall: %f\tscore: %f"%(precision, recall, score))

    pdb.set_trace()




if __name__ == "__main__":
    main()
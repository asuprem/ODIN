import mlep.data_encoder.w2vGeneric as w2vGeneric
import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets



def main():

    traindata = BatchedLocal.BatchedLocal(data_source="./data/initialTrainingData.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    traindata.load()
    _encoder = w2vGeneric.w2vGeneric()
    _encoder.setup(modelPath="w2v-wiki-wikipedia-5000.bin", trainMode="python")

    X_train = _encoder.batchEncode(traindata.getData())
    X_centroid = _encoder.getCentroid(X_train)

    load_data = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)

    while load_data.next():
        _data = load_data.getData()
        _encoded = _encoder.encode(_data)

        _distance = _encoder.getDistance(_encoded, X_centroid)

        print(_distance)








if __name__ == "__main__":
    main()
import mlep.data_encoder.DataEncoder

class bowEncoder(mlep.data_encoder.DataEncoder.DataEncoder):
    """ Built-in encoder for bag of words; """

    def __init__(self,):
        pass

    def setup(self, modelFileName="bow.model"):
        from sklearn.externals import joblib
        from sklearn.metrics.pairwise import cosine_similarity
        from numpy import squeeze, asarray
        import os
        self.squeeze =  squeeze
        self.asarray = asarray
        self.cosine_similarity = cosine_similarity

        modelFilePath = "./Sources/" + modelFileName
        if not os.path.exists(modelFilePath):
            raise IOError(modelFilePath + " not found.")
        self.model = joblib.load(modelFilePath)
        
    def encode(self, data):
        """ data MUST be a list of string """
        try:
            return self.model.transform(data)
        except ValueError:
            return self.model.transform([data])

    def batchEncode(self, data):
        """ batch encode. data must be a list of stringds"""
        return self.model.transform(data)

    def failCondition(self,rawFileName="bow.txt", modelFileName="bow.model"):
        
        bowFilePath = "./RawSources/" + rawFileName

        from sklearn.feature_extraction.text import CountVectorizer
        with open(bowFilePath, 'r') as bow_file:
            bow_doc = bow_file.read()
        
        #Create the vectorizer and fit the bow document to create the vocabulary
        vectorizer = CountVectorizer()
        vectorizer.fit([bow_doc])
        
        # Save the Encoder
        from sklearn.externals import joblib
        joblib.dump(vectorizer, "./Sources/"+modelFileName)
        return True        

        #self.transformed_data = vectorizer.transform(self.source_data['text'].values)

    def getCentroid(self,data):
        # Need all this fancy stuff because Vectorizer returns a matrix
        return 1.0 - self.squeeze(self.asarray(data.mean(axis=0)))

        #return data.mean(axis=0)

    def getDistance(self, queryPoint, centroid):
        return self.cosine_similarity(self.asarray(queryPoint.todense()), centroid.reshape(1,-1)).mean()
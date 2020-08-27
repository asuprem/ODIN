import mlep.data_encoder.DataEncoder

class w2vGoogleNews(mlep.data_encoder.DataEncoder.DataEncoder):
    """ Built-in encoder for Google w2v; limited to 100K most common words """

    def __init__(self,):
        pass

    def setup(self,):
        from gensim.models import KeyedVectors  # pylint: disable=import-error
        from gensim.utils import tokenize   # pylint: disable=import-error
        from numpy import zeros

        from sklearn.metrics.pairwise import cosine_similarity
        from numpy import squeeze, asarray
        
        self.squeeze =  squeeze
        self.asarray = asarray
        self.cosine_similarity = cosine_similarity


        self.model = KeyedVectors.load_word2vec_format('./Sources/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore', limit=100000)
        self.zeros = zeros
        self.zero_v = self.zeros(shape=(300,))
        self.tokenize = tokenize

    def encode(self, data):
        """ data MUST be a string """
        tokens = list(self.tokenize(data))
        # this is for possibly empty tokens
        transformed_data = self.zeros(shape=(300,))
        if not tokens:
            pass
        else:
            for word in tokens:
                transformed_data += self.model[word] if word in self.model else self.zero_v
            transformed_data/=len(tokens)
        return transformed_data



    def batchEncode(self, data):
        """ batch encode. data must be a list of stringds"""
        max_len = len(data)
        transformed_data = self.zeros(shape=(max_len,300))
        
        for idx, sentence in enumerate(data):
            transformed_data[idx] = self.encode(sentence)
        return transformed_data

    def failCondition(self,):
        # no idea how to handle fail condition here
        raise NotImplementedError()

    def getCentroid(self,data):
        # Need all this fancy stuff because Vectorizer returns a matrix
        return data.mean(axis=0)

    def getDistance(self, queryPoint, centroid):
        """ Compute cosine similarity 
        
        The result is in [0,1]. After inversion (1-*), a value closer to 0 means similar, while a value closer to 1 means dissimilar.
        
        """

        return 1.0 - self.cosine_similarity(queryPoint.reshape(1,-1), centroid.reshape(1,-1)).mean()
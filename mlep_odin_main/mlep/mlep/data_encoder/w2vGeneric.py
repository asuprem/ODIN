import mlep.data_encoder.DataEncoder

class w2vGeneric(mlep.data_encoder.DataEncoder.DataEncoder):
    """ Built-in encoder for Generic w2v;"""

    def __init__(self,):
        pass
    
    def setup(self,modelPath = "GoogleNews-vectors-negative300.bin", trainMode = "C", binary=True, unicode_errors = "ignore", limit=None ):
        """
            modelPath -- Name of the w2v model
            trainMode -- ["python","C"]
                            Since this is a gensim wrapper, training mode is required to know how to load the modelfile
            binary -- [True, False]; whether the model is a binary file or non-binary file
            unicode_errors -- ["ignore", "strict"]
            limit -- [None, int]
                        How many words to keep. None means all words are retained

            All options after trainMode are only relevant if trainMode is C

        """
        from gensim.models import KeyedVectors  # pylint: disable=import-error
        from gensim.utils import tokenize   # pylint: disable=import-error
        from numpy import zeros 

        from sklearn.metrics.pairwise import cosine_similarity
        from numpy import squeeze, asarray
        import os

        self.squeeze =  squeeze
        self.asarray = asarray
        self.cosine_similarity = cosine_similarity

        self.zeros = zeros
        self.zero_v = self.zeros(shape=(300,))
        self.tokenize = tokenize
        
        self.modelPath = "./Sources/" + modelPath
        if not os.path.exists(self.modelPath):
            raise IOError(self.modelPath + " not found.")
        if trainMode == "C":
            self.model = KeyedVectors.load_word2vec_format(self.modelPath, binary=binary, unicode_errors=unicode_errors, limit=limit)
        else:
            self.model = KeyedVectors.load(self.modelPath)
        
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

    def failCondition(self,dimensionSize="5000", seedName="wikipedia"):

        # Check if model already exists
        modelSaveName = "-".join(["w2v","wiki", str(seedName), str(dimensionSize)]) + ".bin"
        modelSavePath = "./Sources/"+modelSaveName
        import os
        if os.path.exists(modelSavePath):
            return True
        else:
            # Check if rwa file already exists. If not, create it.
            wikipages = self.getWikipages(dimensionSize, seedName)
            
            import gensim   # pylint: disable=import-error
            #get tokenized forms
            documents = [gensim.utils.simple_preprocess(item) for item in wikipages]
            model = gensim.models.Word2Vec(documents, size=300, window=10,min_count=2,workers=10)
            model.train(documents, total_examples=len(documents),epochs=10)
            model.save(modelSavePath)
            return True

    def getCentroid(self,data):
        # Need all this fancy stuff because Vectorizer returns a matrix
        return data.mean(axis=0)

    def getDistance(self, queryPoint, centroid):
        """ Compute cosine similarity 
        
        The result is in [0,1]. After inversion (1-*), a value closer to 0 means similar, while a value closer to 1 means dissimilar.
        
        """
        return 1.0 - self.cosine_similarity(queryPoint.reshape(1,-1), centroid.reshape(1,-1)).mean()


    def getWikipages(self,dimensionSize, seed):
        import os, pickle       
        
        
        wikipagesFileName = str(seed) + '_' + str(dimensionSize) +'.wikipages'
        wikipagesFilePath = os.path.join('./RawSources',wikipagesFileName)
        listOfWikiPages=[]
        listOfWikiTitles={}
        
        if not os.path.exists(wikipagesFilePath):
            import random
            import wikipedia    # pylint: disable=import-error
            random.seed(a=seed)

            articleTitleNames=[]
            wikiTitlesFileName = 'enwiki-latest-all-titles-in-ns0'
            wikiTitlesFilePath = os.path.join('./RawSources',wikiTitlesFileName)
            with open(wikiTitlesFilePath, 'r') as wikiTitlesFile:
                for line in wikiTitlesFile:
                    if line.startswith('!')  or line.startswith('`'):
                        continue
                    articleTitleNames.append(line.strip())
            
            #Now articleTitleNames has list of wikipedia titles. We have to download these, and create td-idf matrix from their texts
            while len(listOfWikiPages) < int(dimensionSize):
                _title = random.randint(0, len(articleTitleNames)-1)
                _title = articleTitleNames[_title].strip()
                if _title in listOfWikiTitles:
                    continue
                wiki_text = wikipedia.page(_title)
                if len(wiki_text.content.split(' ')) < 100:
                        continue
                listOfWikiTitles[_title] = 1
                listOfWikiPages.append(wiki_text.content)
            with open(wikipagesFilePath, 'wb') as wikipagesFileName:
                pickle.dump([listOfWikiPages, listOfWikiTitles], wikipagesFileName)
        else:
            with open(wikipagesFilePath, 'rb') as wikipagesFileName:
                wiki_data = pickle.load(wikipagesFileName)
                listOfWikiPages = wiki_data[0]
                listOfWikiTitles = wiki_data[1]
        return listOfWikiPages
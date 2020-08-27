import mlep.data_encoder.DataEncoder
import mlep.models.vaegan_model_builder
import torch

class dagan(mlep.data_encoder.DataEncoder.DataEncoder):
    """ Built-in encoder for Generic w2v;"""

    def __init__(self,):
        pass
    
    def setup(self,modelPath = "model.pth", ):
        """
            modelPath -- path to model

        """
        from numpy import zeros 
        from sklearn.metrics.pairwise import cosine_similarity
        from numpy import squeeze, asarray
        import os

        self.modelPath = "./Sources/" + modelPath
        if not os.path.exists(self.modelPath):
            raise IOError(self.modelPath + " not found.")

        self.model = mlep.models.vaegan_model_builder(arch="VAEGAN", base=32, \
                                latent_dimensions = 128, \
                                channels=3)
        self.model.load_state_dict(torch.load(modelPath))
        self.model.cuda()
        self.model.eval()

        self.squeeze =  squeeze
        self.asarray = asarray
        self.cosine_similarity = cosine_similarity

        self.zeros = zeros
        self.zero_v = self.zeros(shape=(300,))
        #self.tokenize = tokenize
        
        
    def encode(self, data): #Should be a torch tensor...
        #tokens = list(self.tokenize(data))
        # this is for possibly empty tokens
        encoded_data = self.model.Encoder(torch.tensor(data).cuda()).squeeze()
        return encoded_data



    def batchEncode(self, data):
        """ data must be a list of tensors"""
        max_len = len(data)
        transformed_data = self.zeros(shape=(max_len,256))
        
        for idx, data_ in enumerate(data):
            transformed_data[idx] = self.model.Encoder(torch.tensor(data_).cuda()).squeeze()
        return transformed_data

    def failCondition(self,*args, **kwargs):

        raise NotImplementedError

    def getCentroid(self,data):
        # Need all this fancy stuff because Vectorizer returns a matrix
        return data.mean(axis=0)

    def getDistance(self, queryPoint, centroid):
        """ Compute cosine similarity 
        
        The result is in [0,1]. After inversion (1-*), a value closer to 0 means similar, while a value closer to 1 means dissimilar.
        
        """
        return 1.0 - self.cosine_similarity(queryPoint.reshape(1,-1), centroid.reshape(1,-1)).mean()
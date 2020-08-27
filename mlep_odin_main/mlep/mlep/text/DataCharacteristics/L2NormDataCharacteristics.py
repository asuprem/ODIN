import numpy as np
import mlep.tools.distributions.DistanceDistribution as DistanceDistribution
import mlep.utils.array_utils as array_utils

class L2NormDataCharacteristics:
    def __init__(self, nBins=40, alpha = 0.6):
        self.nBins = nBins
        self.distribution = None
        self.alpha = alpha
        
        self.attrs={}
        self.attrs["centroid"] = None
        self.attrs["delta_low"] = None
        self.attrs["delta_high"] = None
        
        
    def buildDistribution(self,centroid,data):
        """ Build the data distribution given a data set and its centroid.

        Uses cosine_similary metric to build the characteristics 'map' of a text-based dataset

        """

        self.attrs["centroid"] = centroid

        
        self.distribution = [0]*data.shape[0]
        for idx in range(data.shape[0]):
            self.distribution[idx] = np.linalg.norm(data[idx]-centroid)
        self.distribution = DistanceDistribution.DistanceDistribution(self.nBins, self.distribution)

        self.delta_low_index, self.delta_high_index, _ = array_utils.getSubArray(self.distribution.dist[1:-1], self.nBins-2, int(self.alpha*data.shape[0]))
        self.delta_high_index+=1
        self.delta_low_index+=1
        self.attrs["delta_low"] = self.distribution.dist_keys[self.delta_low_index] - (1./self.nBins)
        self.attrs["delta_high"] = self.distribution.dist_keys[self.delta_high_index]


    def get(self,_key):
        return self.attrs[_key]
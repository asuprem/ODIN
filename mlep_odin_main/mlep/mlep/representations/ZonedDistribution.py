import mlep.tools.distributions.CosineSimilarityDistribution as CosineSimilarityDistribution
import mlep.tools.metrics.TextMetrics as TextMetrics
import mlep.utils.array_utils as array_utils

class ZonedDistribution:
    def __init__(self, nBins=40, alpha = 0.6, metric_callback=None, distribution_callback=None):

        if distribution_callback is None:
            raise ValueError("Must provide a distribution callback.")
        if metric_callback is None:
            raise ValueError("Must provide metric_callback to calculate distance")
        self.distribution_callback = distribution_callback
        self.metric_callback=metric_callback
        
        self.nBins = nBins
        self.distribution = None
        self.alpha = alpha

        self.attrs={}
        self.attrs["centroid"] = None
        self.attrs["delta_low"] = None
        self.attrs["delta_high"] = None
        

    def build(self,centroid,data):
        """ Build the data distribution given a data set and its centroid.

        Uses cosine_similary metric to build the characteristics 'map' of a text-based dataset

        """

        self.attrs["centroid"] = centroid
        self.distribution = [0]*data.shape[0]
        for idx in range(data.shape[0]):
            self.distribution[idx] = self.metric_callback(centroid,data[idx])
        self.distribution = self.distribution_callback(self.nBins, self.distribution)
        
        # Now that distribution is set up, we need to obtain the Data characteristics, namely the concentration of points...
        #self.max_peak_key = self.distribution.dist.index(max(self.distribution.dist))
        # need from 1:-1 because the outside ones are "outliers"...
        self.delta_low_index, self.delta_high_index, _ = array_utils.getSubArray(self.distribution.dist[1:-1], self.nBins-2, int(self.alpha*data.shape[0]))
        self.delta_high_index+=1
        self.delta_low_index+=1
        self.attrs["delta_low"] = self.distribution.dist_keys[self.delta_low_index] - (1./self.nBins)
        self.attrs["delta_high"] = self.distribution.dist_keys[self.delta_high_index]
 
    
    def get(self,_key):
        return self.attrs[_key]
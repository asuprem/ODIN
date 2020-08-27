
import numpy as np, bisect


class CosineSimilarityDistribution:
    """ Histogram based on cosine similarity.

    Attributes:
        dist -- The values at each index in the distribution
        dist_keys -- The keys of the distribution, which are the actual values for each bin (upper range of the bin) Values outside the distribution go in the final bin.
        max_len -- Number of items in the distribution
    """

    def __init__(self, nBins=40, data=None):
        """ Initializes the distribution.

        One needs to provide the number of bins for the distribution. (default is 40 bins). If data is provided, distribution is also built.

        Args:
            nBins -- INT. Number of bins in the distribution
            data -- array nx1. Builds the distribution
        """
        if data is None:
            raise ValueError("data not provided. DistanceDistribution requires some data during construction.")
        self.dat_min = 0.0
        self.dat_max = 1.0
        self.nBins = nBins
        self.dist = [0 for _ in range(self.nBins+1)]
        self.dist_keys = np.linspace(self.dat_min, self.dat_max, self.nBins).tolist()
        self.max_len = 0.0
        self.build(data)


    def _findIndex(self,data):
        i = bisect.bisect(self.dist_keys, data)
        return i

    def get(self,data):
        """ Get the bin index of a new data point from the existing distribution. 

        Args:
            data -- A data item. In this case, a cosine similarity metric (in 0 and 1)

        """
        return self.dist[self._findIndex(data)]/self.max_len

    def update(self,data):
        """ Update the distribution with new data.

        This adds new information to the distribution, updating the components.

        Args:
            data -- A data item. In this casee a cosine similarity metric (in 0 and 1)

        """
        self.max_len+=1.0
        self.dist[self._findIndex(data)] += 1
    
    def build(self,data):
        """ Build a distribution given a bunch of data 

        This initially builds the distribution using data. 

        Args:
            data -- A n x 1 array of data items. 

        """
        for _row in data:
            self.update(_row)

        
import numpy as np
import scipy.spatial.distance

def L2Norm(a,b):
    """ Returns the L2 Norm - basically a wrapper around the numpy linalg method that conforms to what MLEP expects in a NumericMetric
     
    Args:
        a -- INPUT 1
        b -- INPUT 2
    
    """
    return np.linalg.norm(a-b)

def Manhattan(a,b):
    """ Returns the Manhattan Norm (L1 Norm, as otherwise known) - basically a wrapper around the scipy citiblock method that conforms to what MLEP expects in a NumericMetric
     
    Args:
        a -- INPUT 1
        b -- INPUT 2
    
    """
    return scipy.spatial.distance.cityblock(a,b)
	
def BrayCurtis(a,b):
    """ Returns the BrayCurtis Norm - basically a wrapper around the scipy BrayCurtis method that conforms to what MLEP expects in a NumericMetric
     
    Args:
        a -- INPUT 1
        b -- INPUT 2
    
    """
    return scipy.spatial.distance.braycurtis(a,b)

def Canberra(a,b):
	""" Returns the Canberra Norm - basically a wrapper around the scipy Canberra method that conforms to what MLEP expects in a NumericMetric
	 
	Args:
		a -- INPUT 1
		b -- INPUT 2

	"""
	return scipy.spatial.distance.canberra(a,b)

def Chebyshev(a,b):
	""" Returns the Chebyshev Norm - basically a wrapper around the scipy Chebyshev method that conforms to what MLEP expects in a NumericMetric
	 
	Args:
		a -- INPUT 1
		b -- INPUT 2

	"""
	return scipy.spatial.distance.chebyshev(a,b)

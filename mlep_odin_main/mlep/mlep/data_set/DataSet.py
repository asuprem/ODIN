__metaclass__ = type

class DataSet:
    """The DataSet Model is passed into LearningModel for training a new model. Each element is a piece of Data - another class
    It contains methods for calculating distance (from a random data point), etc. It includes encoding methods - link to an existing encoder (?)
    
    other methods - label of a Data point
                    content of a Data point (the encoded data)
                    other attributes of a Data point
                    whether this is regression, multiclass, or binary
                    """

    def __init__(self):
        pass
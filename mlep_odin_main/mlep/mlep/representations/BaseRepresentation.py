__metaclass__ = type

class BaseRepresentation:
    def __init__(self,):
        """ Initialize a representation, with params. """
        raise NotImplementedError()

    def build(self,):
        """ Build the representation. """
        raise NotImplementedError()

    def query(self,):
        """ Returns a representation-specific return with information. """
        raise NotImplementedError()
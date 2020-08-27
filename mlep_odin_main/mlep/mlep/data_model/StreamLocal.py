import mlep.data_model.DataModel

class StreamLocal(mlep.data_model.DataModel.DataModel):
    """ StreamLocal model. Simulates streaming data"""
    
    def __init__(self, data_source=None, data_mode=None, data_set_class=None):
        """
        Initializes a StreamLocal reader. 
        
        StreamLocal  reads a file locally as a stream, delivering data one piece at a time.

        Args:
            data_source: Source of the data to read.
            data_mode: "single" or "split". "split" is not implemented. "single" means there is a single file with all classes present together.
            data_set_class: A dataset class for each line read from the file or each file read (if not 'single' mode). The dataset class must initialize with only the data item. It must implement the getData() and the getLabel() functions. getData() must not return None.

        """

        # Init function loads data
        self._object = None

        self.data_source = data_source
        if data_mode == "single":
            self.reader = open(self.data_source, "r")
        else:
            raise NotImplementedError()            
        self._idx=0
        self.data_set_class = data_set_class


    def getData(self,):
        """ Gets the data item 

        Returns:
            mlep.data_set derived class wrapper's getData() method (self.data_set_class)

        Raises:
            IOError

        """
        if not self._idx:
            raise IOError("Trying to access stream without reading from it. Run next() before accessing data.")
        return self._object.getData()

    def getLabel(self,):
        """ Gets the data label.

        The label can be None if the data item in the stream is an unlabeled one. 

        Returns:
            mlep.data_set derived class wrapper's getLabel() method  (self.data_set_class)

        Raises:
            IOError

        """
        if not self._idx:
            raise IOError("Trying to access stream without reading from it. Run next() before accessing data.")
        return self._object.getLabel()

    def getObject(self,):
        """ Gets the data object itself.

        Returns:
            mlep.data_set derived class wrapper around the data sample  (self.data_set_class)

        Raises:
            IOError

        """
        if not self._idx:
            raise IOError("Trying to access stream without reading from it. Run next() before accessing data.")
        return self._object
        
    def next(self,):
        """ Iterates to the next item.

        This is just a basic filestream. Future work to include loading data split across multiple files (using, perhaps, some distributed load paradigm?)

        Raises:
            ValueError

        """
        line = self.reader.readline()
        if line == "":
            # End of file
            self.reader.close()
            return False
        
        self._object = self.data_set_class(line)

        if self._object.getData() is None:
            raise ValueError("Data is NoneType at line %i in %s"%(self._idx+1,self.data_source))

        self._idx+=1

        return True
        
    def streamLength(self):
        """ Returns number of items read

        Returns:
            self._idx

        """
        return self._idx

    def getNextBatchData(self,):
        raise NotImplementedError()

    def getNextBatchLabels(self,):
        raise NotImplementedError()






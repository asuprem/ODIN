

class MemoryTracker:
    """ This is a MemoryTracker class to track data during classification 
    
    Attributes:
        MEMORY_TRACKER -- dictionary of Memory Objects (only BatchedLocal implemented)
        MEMORY_MODE -- "default" only. predetermined data_model and data_set_class
        CLASSIFY_MODE -- "binary" only. Not set up to handle multiclass data
        MEMORY_STORE -- dictionary. Storage mode for each Memory object ("local" or ""in-memory". Only in-memory implemented)

    """
    def __init__(self,memory_mode="default"):
        """ 
        Initializes the MemoryTracker

        Args:
            memory_mode -- "default": data_set_class is PseudoJsonTweets. Placeholder for future

        """
        self.MEMORY_TRACKER= {}
        self.MEMORY_MODE = memory_mode
        self.CLASSIFY_MODE = "binary"
        self.MEMORY_STORE = {}
    
    def addNewMemory(self,memory_name, memory_store="local", memory_path = None):
        """ 
        Adds a new memory

        Args:
            memory_name -- Name of the memory
            memory_store -- "local": memory will be saved to disk. "memory" -- memory will be in-memory. Not Implemented
            memory_path -- folder where memory will be stored and loaded from is memory_store is "local"

        Raises:
            ValueError
        """
        import os
        if memory_name in self.MEMORY_TRACKER:
            raise ValueError("memory_name: %s    already exists in this memory" % memory_name)
        if memory_store == "local" and memory_path is None:
            raise ValueError("Must provide memory_path if using 'local' memory_store.")
        
        if self.MEMORY_MODE == "default":
            if memory_store == "local":
                from mlep.data_model.BatchedLocal import BatchedLocal
                from mlep.data_set.PseudoJsonTweets import PseudoJsonTweets
                data_source = memory_name + "_memory.json"
                data_source_path = os.path.join(memory_path, data_source)
                self.MEMORY_TRACKER[memory_name] = BatchedLocal(data_source=data_source_path, data_mode="single", data_set_class=PseudoJsonTweets)
                self.MEMORY_TRACKER[memory_name].open(mode="a")
                self.MEMORY_STORE[memory_name] = memory_store
            elif memory_store == "memory":
                from mlep.data_model.BatchedLocal import BatchedLocal
                from mlep.data_set.PseudoJsonTweets import PseudoJsonTweets
                data_source = []
                #data_source_path = os.path.join(memory_path, data_source)
                self.MEMORY_TRACKER[memory_name] = BatchedLocal(data_source=data_source, data_mode="single", data_set_class=PseudoJsonTweets, data_location="memory")
                self.MEMORY_STORE[memory_name] = memory_store

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def addToMemory(self,memory_name, data):
        """ 
        Add data to memory 
        
        Args:
            memory_name -- name of memory to which data will be added
            data -- data item to add
            
        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """
        # TODO add integrity check -- is data of the same type as memory's datatype???
        self.MEMORY_TRACKER[memory_name].write(data)
    
    def clearMemory(self,memory_name):
        """ 
        Clear memory

        Args:
            memory_name -- name of memory to clear

        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """
        self.MEMORY_TRACKER[memory_name].clear()

    def hasSamples(self,memory_name):
        """ 
        Return whether memory has samples.

        Args:
            memory_name -- name of memory

        Returns:
            bool: True is memory has items. False if memory has no items

        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """

        return self.MEMORY_TRACKER[memory_name].hasSamples()

    def isEmpty(self,memory_name):
        """ 
        Return whether memory is empty.

        Args:
            memory_name -- name of memory

        Returns:
            bool: True is memory is empty items. False if memory has items

        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """

        return not self.hasSamples(memory_name)

    def memorySize(self,memory_name):
        """ 
        Return total number of samples in memory.

        Args:
            memory_name -- name of memory

        Returns:
            INT -- number of samples in memory

        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """

        return self.MEMORY_TRACKER[memory_name].memorySize()

    def getMemoryNames(self,):
        """
        Return all memory names

        Args:
            None

        Returns:
            [list] -- list of memory_names in self.MEMORY_TRACKER
        """
        return [item for item in self.MEMORY_TRACKER]

    def transferMemory(self,memory_name):
        """
        Transfer memory to a copy of the source class of the memory and return the new class. In addition, clear the memory.

        Args:
            memory_name -- name of memory to transfer
        
        Returns
            New class with pointer to memory of self.MEMORY_TRACKER[memory_name]. Only works for "local" memory. TODO Implement for in-memory.

        """
        if self.MEMORY_MODE == "default":
            if self.MEMORY_STORE[memory_name] == "local":
                self.MEMORY_TRACKER[memory_name].close()
                loadModule = self.MEMORY_TRACKER[memory_name].__class__.__module__
                loadClass  = self.MEMORY_TRACKER[memory_name].__class__.__name__
                dataModelModule = __import__(loadModule, fromlist=[loadClass])
                dataModelClass = getattr(dataModelModule, loadClass)
                # Get SCHEDULED_DATA_FILE from MEMORY_TRACK
                return dataModelClass(**self.MEMORY_TRACKER[memory_name].__getargs__())
            elif self.MEMORY_STORE[memory_name] == "memory":
                loadModule = self.MEMORY_TRACKER[memory_name].__class__.__module__
                loadClass  = self.MEMORY_TRACKER[memory_name].__class__.__name__
                dataModelModule = __import__(loadModule, fromlist=[loadClass])
                dataModelClass = getattr(dataModelModule, loadClass)
                # Get SCHEDULED_DATA_FILE from MEMORY_TRACK
                return dataModelClass(**self.MEMORY_TRACKER[memory_name].__getargs__())
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def getClassifyMode(self):
        """ 
        Get classify mode for this memory

        Args:
            None

        Returns:
            str -- self.CLASSIFY_MODE

        """
        return self.CLASSIFY_MODE





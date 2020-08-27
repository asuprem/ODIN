import mlep.data_model.DataModel

class BatchedLocal(mlep.data_model.DataModel.DataModel):
    """ BatchedLocal model loads the batchedlocal file"""
    
    def __init__(self, data_source=None, data_mode=None, num_samples="all", data_set_class=None, classification_mode="binary", classes=[0,1], data_location="local"):
        """

        data_source -- path of the Local Batched Data File

        num_samples -- how many samples to load at a time...
                -- also not implemented. For now we load all examples in memory
                -- TODO add option for lazy loading vs preloading
        
        data_mode -- [collected/sequential]
            - "single" -- all samples are in a single file - the one linked in data_source
            - "sequential" - samples are spread out across multiple files. Probably will use some form of wildcarding for this. TBD.
            - "split" - each sample is a file. Each sample must be in their corresponding label folder (or something) - TBD. Sort of like keras data generator


            We only handle collected so far. Batched also means all data is loaded into memory

        data_set_class -- Class file (from DataSet) that encapsulates each example.

        examples are delimited by newlines
        """
        if data_location == "local":
            self.data=[]
            self.class_data={}
            self.class_statistics={}
            self.classes=classes
            self.num_samples = num_samples
            self.classification_mode = classification_mode
            for _class in classes:
                self.class_data[_class] = []
                self.class_statistics[_class] = 0
            
            self.data_mode = data_mode
            self.data_source = data_source
            self.data_set_class = data_set_class
            self.data_location = data_location
        elif data_location == "memory":
            self.data=data_source
            self.class_data={}
            self.class_statistics={}
            self.classes=classes
            self.num_samples = num_samples
            self.classification_mode = classification_mode
            for _class in classes:
                self.class_data[_class] = []
                self.class_statistics[_class] = 0
            self.data_mode = data_mode
            self.data_source = None
            self.data_set_class = data_set_class
            self.data_location = data_location
        else:
            raise NotImplementedError()

        # This is whether, in write mode, we are adding to existing file or clearing and starting over from scratch. Lazy deletion.
        self._clear = True
        self.load_memory = 0
        self.writer = None

    def load(self,):
        # Init function loads data
        # load into a dataframe---?
        
        if self.data_mode == "single":
            if self.data_location == "local":
                with open(self.data_source,"r") as data_source_file:
                    for line in data_source_file:
                        self.data.append(self.data_set_class(line))
            elif self.data_location == "memory":
                pass
            else:
                raise NotImplementedError()

        # so self.data is a list of [data_set_class(), data_set_class()...]  
    def load_by_class(self,):
        if self.data_mode == "single":
            if self.data_location == "local":
                with open(self.data_source,"r") as data_source_file:
                    for line in data_source_file:
                        self.data.append(self.data_set_class(line))
                        if self.data[-1].getLabel() not in self.classes:
                            raise ValueError("Label of data item: " + str(self.data[-1].getLabel()) + " does not match any in " + str(self.classes))
                        self.class_data[self.data[-1].getLabel()].append(self.data[-1])
                        self.class_statistics[self.data[-1].getLabel()] += 1
            elif self.data_location == "memory":
                self.class_data={}
                self.class_statistics={}
                for _class in self.classes:
                    self.class_data[_class] = []
                    self.class_statistics[_class] = 0
                for entry in self.data:
                    # clear first
                    self.class_data[entry.getLabel()].append(entry)
                    self.class_statistics[entry.getLabel()] += 1
            else:
                raise NotImplementedError()
    
    def all_class_sizes(self,):
        return sum([self.class_statistics[item] for item in self.class_statistics])
    def class_size(self,_class):
        return self.class_statistics[_class]


    def open(self, mode="a"):
        if self.writer is not None:
            raise ResourceWarning("Writer is still open. You should close it with close() method before opening again.")
        self.writer = open(self.data_source, mode)
    def close(self,):
        if self.writer is None:
            pass
        else:
            self.writer.close()
        self.writer = None


    # data is a DataSet object
    def write(self,data):
        # mode --> "w" or "a"
        if self.data_location == "local":
            if self.writer is None:
                raise IOError("No writer set.")
            if self.data_mode == "single":
                self.writer.write(data.serialize()+'\n')
                self.load_memory+=1
            else:
                raise NotImplementedError()
            # For other data_mode, have to keep track of file names for writing.
        elif self.data_location == "memory":
            self.data.append(data)
            self.load_memory+=1
        else:
            raise NotImplementedError()

    def clear(self,):
        """ this is to clear the BatchedLocal file. Dangerous for loading data batchedFile object, cause it will, well, delete the load data """
        """ maybe separate BatchedLocalLoader and BatchedLocalWriters..."""
        if self.data_location == "local":
            self._clear = True
            self.close()
            import os
            if os.path.exists(self.data_source):
                os.remove(self.data_source)
            self.load_memory = 0
            self.open("a")
        elif self.data_location == "memory":
            self._clear = True
            self.data = []
            self.load_memory = 0
        else:
            raise NotImplementedError()
    
    def hasSamples(self,):
        return bool(self.load_memory)
    def memorySize(self,):
        return self.load_memory

    
    
    def getData(self,):
        return [dataItem.getData() for dataItem in self.data]

    def getLabels(self,):
        return [dataItem.getLabel() for dataItem in self.data]

    def getObjects(self,):
        return self.data
    
    def augment(self,_objects):
        self.data+=_objects
        

    def getDataByClass(self,_class):
        return [dataItem.getData() for dataItem in self.class_data[_class]]
    def getLabelsByClass(self,_class):
        return [dataItem.getLabel() for dataItem in self.class_data[_class]]
    def getObjectsByClass(self,_class):
        return self.class_data[_class]

    def augment_by_class(self,_objects, _class):
        self.data+=_objects
        self.class_data[_class]+=_objects

        self.class_statistics[_class] += len(_objects)

    # toRemove - how many to remove
    def prune_by_class(self,_class, toRemove):
        from random import sample
        self.class_data[_class] = sample(self.class_data[_class], self.class_statistics[_class]-toRemove)
        self.class_statistics[_class] = len(self.class_data[_class])

        # reset data <-- can be optimized TODO
        self.data = []
        for _class in self.classes:
            self.data += self.class_data[_class]


    def getNextBatchData(self,):
        raise NotImplementedError()

    def getNextBatchLabels(self,):
        raise NotImplementedError()

    def getSource(self):
        return self.data_source
    def getMode(self,):
        return self.data_mode
    def getDataSetClass(self):
        return self.data_set_class

    """ This returns a dictionary of arguments """
    def __getargs__(self,):
        if self.data_location == "local":
            return {
                "data_source":self.data_source,
                "data_mode":self.data_mode,
                "num_samples":self.num_samples,
                "data_set_class":self.data_set_class,
                "classification_mode":self.classification_mode,
                "classes":self.classes,
                "data_location":self.data_location
            }
        elif self.data_location == "memory":
            return {
                "data_source":self.data,
                "data_mode":self.data_mode,
                "num_samples":self.num_samples,
                "data_set_class":self.data_set_class,
                "classification_mode":self.classification_mode,
                "classes":self.classes,
                "data_location":self.data_location
            }
        else:
            raise NotImplementedError()






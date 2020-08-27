import mlep.data_set.DataSet

class BDDData(mlep.data_set.DataSet.DataSet):
    """ BDDData object """

    def __init__(self,data, dataKey, labelKey):
        from json import loads, dumps
        self.dumps = dumps
        self.raw = loads(data)
        self.data = self.raw[dataKey]
        self.labelKey = labelKey
        if self.labelKey in self.raw:
            self.label = self.raw[self.labelKey]
        else:
            self.label = None

    def getData(self,):
        return self.data

    def getLabel(self,):
        return self.label

    def getValue(self,key):
        return self.raw[key]

    def serialize(self,):
        return self.dumps(self.raw)

    def setLabel(self,label):
        self.raw[self.labelKey] = label
        self.label = label


    
import sqlite3
import mlep.utils.sqlite_utils as sqlite_utils

class ModelDB:
    def __init__(self,):
        pass
        self.configureSqlite()
        self.setupDbConnection()
        self.initializeDb()


    def configureSqlite(self):
        """Configure SQLite to convert numpy arrays to TEXT when INSERTing, and TEXT back to numpy
        arrays when SELECTing.
        
        """
        import numpy as np
        sqlite3.register_adapter(np.ndarray, sqlite_utils.adapt_array)
        sqlite3.register_converter("array", sqlite_utils.convert_array)


    def setupDbConnection(self):
        """ Set up connection to a SQLite database. """
        self.DB_CONN = None
        self.DB_CONN = sqlite3.connect("test.db", detect_types=sqlite3.PARSE_DECLTYPES)

    def initializeDb(self):
        """Create tables in a SQLite database."""

        cursor = self.DB_CONN.cursor()
        cursor.execute("""Drop Table IF EXISTS Models""")
        cursor.execute("""
            CREATE TABLE Models(
                modelid         text,
                parentmodel     text,
                pipelineName    text,
                timestamp       real,
                data_centroid   array,
                trainingModel   text,
                trainingData    text,
                testData        text,
                precision       real,
                recall          real,
                fscore          real,
                type            text,
                active          integer,
                PRIMARY KEY(modelid),
                FOREIGN KEY(parentmodel) REFERENCES Models(trainingModel)
            )
        """)
        self.DB_CONN.commit()
        cursor.close()

    def close(self,):
        try:
            self.DB_CONN.close()
        except sqlite3.Error:
            pass



    def insertModelToDb(self,modelid=None, parentmodelid=None, pipelineName=None,
                                timestamp=None, data_centroid=None, training_model=None, 
                                training_data=None, test_data=None, precision=None, recall=None, score=None,
                                _type=None, active=1):
        columns=",".join([  "modelid","parentmodel","pipelineName","timestamp","data_centroid",
                                "trainingModel","trainingData","testData",
                                "precision","recall","fscore",
                                "type","active"])
            
        sql = "INSERT INTO Models (%s) VALUES " % columns
        sql += "(?,?,?,?,?,?,?,?,?,?,?,?,?)"
        cursor = self.DB_CONN.cursor()
        
        cursor.execute(sql, (   modelid,
                                parentmodelid,
                                pipelineName, 
                                timestamp,
                                data_centroid,
                                training_model,
                                training_data,
                                test_data,
                                precision,
                                recall,
                                score,
                                _type,
                                active) )
        
        self.DB_CONN.commit()
        cursor.close()

    def getModelsSince(self, _time = None):
        cursor = self.DB_CONN.cursor()
        if _time is None:
            # We are getting ALL models
            sql = "select trainingModel from Models"
        else:
            # We are getting models since a time
            sql = "select trainingModel from Models where timestamp > %s" % _time
        
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        return [item[0] for item in tupleResults]

    def getNewModelsSince(self, _time = None):
        cursor = self.DB_CONN.cursor()
        if _time is None:
            # We are getting ALL models
            sql = "select trainingModel from Models where parentmodel IS NULL"
        else:
            # We are getting models since a time
            sql = "select trainingModel from Models where timestamp > %s and parentmodel IS NULL" % _time
        
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        return [item[0] for item in tupleResults]
        
        
    def getUpdateModelsSince(self, _time = None):
        cursor = self.DB_CONN.cursor()
        if _time is None:
            # We are getting ALL models
            sql = "select trainingModel from Models where parentmodel IS NOT NULL"
        else:
            # We are getting models since a time
            sql = "select trainingModel from Models where timestamp > %s and parentmodel IS NOT NULL" % _time
        
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        return [item[0] for item in tupleResults]

    def getModelDetails(self,ensembleModelNames, toGet = None):
        cursor = self.DB_CONN.cursor()
        if toGet is None:
            toGet = ["trainingModel","fscore","pipelineName"]
        sql = "select " + ",".join(toGet) + " from Models where trainingModel in ({seq})".format(seq=",".join(["?"]*len(ensembleModelNames)))
        cursor.execute(sql,ensembleModelNames)
        tupleResults = cursor.fetchall()
        cursor.close()
        dictResults = {}
        for entry in tupleResults:
            dictResults[entry[0]] = {}
            for idx,val in enumerate(toGet):
                dictResults[entry[0]][val] = entry[idx]
        return dictResults

    def getDetails(self,dataDict,keyVal,_format, order=None):
        if _format == "list":
            if order is None:
                # We need the order for lists
                raise RuntimeError("No order provided for getDetails with 'list' format")
            details = []
            details = [dataDict[item][keyVal] for item in order]
            return details
        if _format == "dict":
            details = {item:dataDict[item][keyVal] for item in dataDict}
            return details

    def getPipelineDetails(self,):
        cursor = self.DB_CONN.cursor()
        sql = "select pipelineName, trainingModel, fscore from Models"
        cursor.execute(sql)
        tupleResults = cursor.fetchall()
        cursor.close()
        dictResults = {}
        for entry in tupleResults:
            if entry[0] not in dictResults:
                dictResults[entry[0]] = []
            # entry[0] --> pipelineName
            # entry[1] --> trainingModel        item[0] in step 3 upon return
            # entry[2] --> fscore               item[1] in step 3 upon return
            dictResults[entry[0]].append((entry[1], entry[2]))
        return dictResults




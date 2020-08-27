import os
import re
import glob

class VehicleIDDataCrawler:
    def __init__(self,data_folder="VehicleID", train_folder="image", test_folder="", query_folder="", **kwargs):
        self.metadata = {}

        self.data_folder = data_folder
        self.train_folder = os.path.join(self.data_folder, train_folder)
        #self.test_folder = os.path.join(self.data_folder, test_folder)
        #self.query_folder = os.path.join(self.data_folder, query_folder)

        self.train_list = "train_list.txt"
        self.query_list = kwargs.get("test_list","test_list_13164") + ".txt"

        list_folder = os.path.join(self.data_folder, "train_test_split")
        self.train_list = os.path.join(list_folder, self.train_list)
        self.query_list = os.path.join(list_folder, self.query_list)

        self.logger = kwargs.get("logger")

        self.__verify(self.data_folder)
        self.__verify(self.train_folder)


        self.crawl()

    def __verify(self,folder):
        if not os.path.exists(folder):
            raise IOError("Folder {data_folder} does not exist".format(data_folder=folder))
        else:
            self.logger.info("Found {data_folder}".format(data_folder = folder))

    def crawl(self,):
        self.metadata["train"], self.metadata["test"], self.metadata["query"] = {}, {}, {}
        self.metadata["train"]["crawl"], self.metadata["train"]["pids"], self.metadata["train"]["cids"], self.metadata["train"]["imgs"] = self.__crawl(self.train_list, reset_labels=True)
        
        self.__querycrawl(self.query_list)

        self.logger.info("Train\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["train"]["pids"], self.metadata["train"]["cids"], self.metadata["train"]["imgs"]))
        self.logger.info("Test \tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["test"]["pids"], self.metadata["test"]["cids"], self.metadata["test"]["imgs"]))
        self.logger.info("Query\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["query"]["pids"], self.metadata["query"]["cids"], self.metadata["query"]["imgs"]))

    def __crawl(self,train_file, reset_labels=False):
        crawler = []
        pids, cids = {}, []
        pid_label = 0
        with open(train_file,"r") as train_:
            for line in train_:
                ln = line.strip().split(" ")
                img_path = ln[0]+".jpg"
                img_path = os.path.join(self.train_folder, img_path)
                pid = int(ln[1])
                cid = 0
                if pid not in pids:
                    pids[pid] = pid_label if reset_labels else pid
                    pid_label += 1
                pids.append(pid)
                cids.append(cid)
                crawler.append((img_path, pids[pid], cid))
        return crawler, len(set(pids.keys())), len(set(cids)), len(crawler)

    def __querycrawl(self,query_file, reset_labels=False):
        crawler = []
        pids, cids = {}, []
        pid_label = 0
        with open(query_file,"r") as train_:
            for line in train_:
                ln = line.strip().split(" ")
                img_path = ln[0]+".jpg"
                img_path = os.path.join(self.train_folder, img_path)
                pid = int(ln[1])
                cid = 0
                if pid not in pids:
                    pids[pid] = pid_label if reset_labels else pid
                    pid_label += 1
                pids.append(pid)
                cids.append(cid)
                crawler.append((img_path, pids[pid], cid))
        
        pid_in_query = {}
        self.metadata["test"]["crawl"], self.metadata["query"]["crawl"] = []


        for crawled_img in crawler:
            img_path, pid, cid = crawled_img
            # check if pid already captured. If so add to gallery. Else add to query
            if pid in pid_in_query:
                self.metadata["test"]["crawl"].append((img_path, pid, cid))
            else:
                pid_in_query[pid] = 1
                self.metadata["query"]["crawl"].append((img_path, pid, cid))
        
        self.metadata["test"]["pids"], self.metadata["test"]["cids"] = len(pids), 1
        self.metadata["query"]["pids"], self.metadata["query"]["cids"] = len(pids), 1
        
        self.metadata["test"]["imgs"] = len(self.metadata["test"]["crawl"])
        self.metadata["query"]["imgs"] = len(self.metadata["query"]["crawl"])
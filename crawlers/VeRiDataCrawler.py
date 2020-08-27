import os
import re
import glob

class VeRiDataCrawler:
  """ Data crawler for the VeRi-776 dataset

    The VeRiDataCrawler crawls the VeRi-776 folder to populate training, query, and gallery sets with images and their respective PIDs and CIDs.

    Args:
      data_folder (str): Name of the VeRi folder. Default: "VeRi"
      train_folder (str): Folder inside data_folder with training images. Default: "image_train"
      test_folder (str): Folder inside data_folder with testing/gallery images. Default: "image_test"
      query_folder (str): Folder inside data_folder with query images. Default: "image_query"
    
    Kwargs:
      logger: Instance of Logging object

    Attributes:
      metadata (dict): Contains image paths, PIDs, and CIDs of training, testing, and query sets
        train (dict): contains image paths, PIDs, and CIDs
          crawl (list): List of tuples. Each tuple is a 3-tuple of (image_path, PID, CID)
          pids (int): Number of PIDs in training set
          cids (int): Number of CIDs in training set
          imgs (int): Number of images in training set
        test (dict): contains image paths, PIDs, and CIDs
          crawl (list): List of tuples. Each tuple is a 3-tuple of (image_path, PID, CID)
          pids (int): Number of PIDs in testing set
          cids (int): Number of CIDs in testing set
          imgs (int): Number of images in testing set
        query (dict): contains image paths, PIDs, and CIDs
          crawl (list): List of tuples. Each tuple is a 3-tuple of (image_path, PID, CID)
          pids (int): Number of PIDs in query set
          cids (int): Number of CIDs in query set
          imgs (int): Number of images in query set

    Methods:
      crawl(): Populate self.metadata
      __verify(folder): Check if folder exists

    """
  def __init__(self,data_folder="VeRi", train_folder="image_train", test_folder="image_test", query_folder="image_query", **kwargs):
    self.metadata = {}

    self.data_folder = data_folder
    self.train_folder = os.path.join(self.data_folder, train_folder)
    self.test_folder = os.path.join(self.data_folder, test_folder)
    self.query_folder = os.path.join(self.data_folder, query_folder)

    self.logger = kwargs.get("logger")

    self.__verify(self.data_folder)
    self.__verify(self.train_folder)
    self.__verify(self.test_folder)
    self.__verify(self.query_folder)

    self.crawl()

  def __verify(self,folder):
    if not os.path.exists(folder):
      raise IOError("Folder {data_folder} does not exist".format(data_folder=folder))
    else:
      self.logger.info("Found {data_folder}".format(data_folder = folder))

  def crawl(self,):
    self.metadata["train"], self.metadata["test"], self.metadata["query"] = {}, {}, {}
    self.metadata["train"]["crawl"], self.metadata["train"]["pids"], self.metadata["train"]["cids"], self.metadata["train"]["imgs"] = self.__crawl(self.train_folder, reset_labels=True)
    self.metadata["test"]["crawl"], self.metadata["test"]["pids"], self.metadata["test"]["cids"], self.metadata["test"]["imgs"] = self.__crawl(self.test_folder)
    self.metadata["query"]["crawl"], self.metadata["query"]["pids"], self.metadata["query"]["cids"], self.metadata["query"]["imgs"] = self.__crawl(self.query_folder)

    self.logger.info("Train\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["train"]["pids"], self.metadata["train"]["cids"], self.metadata["train"]["imgs"]))
    self.logger.info("Test \tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["test"]["pids"], self.metadata["test"]["cids"], self.metadata["test"]["imgs"]))
    self.logger.info("Query\tPIDS: {:6d}\tCIDS: {:6d}\tIMGS: {:8d}".format(self.metadata["query"]["pids"], self.metadata["query"]["cids"], self.metadata["query"]["imgs"]))

  def __crawl(self,folder, reset_labels=False):
    imgs = glob.glob(os.path.join(folder, "*.jpg"))
    _re = re.compile(r'([\d]+)_[a-z]([\d]+)')
    pid_labeler = 0
    pid_tracker, cid_tracker = {}, {}
    crawler = []
    pid_counter, cid_counter, img_counter = 0, 0, 0
    for img in imgs:
      pid, cid = map(int, _re.search(img).groups()) # _re.search lol
      if pid < 0: continue  # ignore junk
      if cid < 0: continue  # ignore junk
      if pid not in pid_tracker:
        pid_tracker[pid] = pid_labeler if reset_labels else pid
        pid_labeler += 1
      if cid not in cid_tracker:
        cid_tracker[cid] = cid-1
      crawler.append((img, pid_tracker[pid], cid-1))  # cids start at 1 in data
    return crawler, len(pid_tracker), len(cid_tracker), len(crawler)
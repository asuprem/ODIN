import torch
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import defaultdict
import random
import os.path as osp
import numpy as np

import pdb
class TDataSet(TorchDataset):
  def __init__(self,dataset, transform, valid_labels):
    self.valid_labels=valid_labels
    self.dataset = [item for item in dataset if item[1] in self.valid_labels]
    self.transform = transform
    
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,idx):
    img, pid, cid = self.dataset[idx]
    img_arr = self.transform(self.load(img))
    return img_arr, pid, idx
  
  def load(self,img):
    if not osp.exists(img):
      raise IOError("{img} does not exist in path".format(img=img))
    img_load = Image.open(img).convert('RGB')
    return img_load

class TSampler(Sampler):
  """ Triplet sampler """
  def __init__(self, dataset, batch_size, instance):
    self.dataset = dataset
    self.batch_size=batch_size
    self.instance = instance
    self.unique_ids = self.batch_size // self.instance
    self.indices = defaultdict(list)
    self.pids = set()
    for idx, (img, pid, cid) in enumerate(self.dataset):
      self.indices[pid].append(idx)
      self.pids.add(pid)
    self.pids = list(self.pids)
    self.batch = 0
    for pid in self.pids:
      num_ids = len(self.indices[pid])
      num_ids = self.instance if num_ids < self.instance else num_ids
      self.batch += num_ids - num_ids % self.instance
  
  def __iter__(self):
    batch_idx = defaultdict(list)
    for pid in self.pids:
      ids = [item for item in self.indices[pid]]
      if len(ids) < self.instance:
        ids = np.random.choice(ids, size=self.instance, replace=True)
      random.shuffle(ids)
      batch, batch_counter = [], 0
      for idx in ids:
        batch.append(idx)
        batch_counter += 1
        if len(batch) == self.instance:
          batch_idx[pid].append(batch)
          batch = []
    _pids, r_pids = [item for item in self.pids], []
    # Optimize this???
    to_remove = {}
    pid_len = len(_pids)
    while pid_len >= self.unique_ids:
      sampled = random.sample(_pids, self.unique_ids)
      for pid in sampled:
        batch = batch_idx[pid].pop(0)
        r_pids.extend(batch)
        if len(batch_idx[pid]) == 0:
          to_remove[pid] = 1

      _pids = [item for item in _pids if item not in to_remove]
      pid_len = len(_pids)
      to_remove = {}

    self.__len = len(r_pids)
    return iter(r_pids)
  
  def __len__(self):
    return self.__len
class Cars196Generator:
  def __init__(self,gpus, i_shape = (208,208), normalization_mean = 0.5, normalization_std = 0.5, normalization_scale = 1./255., h_flip = 0.5, t_crop = True, rea = True, **kwargs):
    """ Data generator for training and testing.

    Args:
      gpus (int): Number of GPUs
      i_shape (int, int): 2D Image shape
      normalization_mean (float): Value to pass as mean normalization parameter to pytorch Normalization
      normalization_std (float): Value to pass as std normalization parameter to pytorch Normalization
      normalization_scale (float): Value to pass as scale normalization parameter. Not used.
      h_flip (float): Probability of horizontal flip for image
      t_crop (bool): Whether to include random cropping
      rea (bool): Whether to include random erasing augmentation (at 0.5 prob)
    
    """
    self.gpus = gpus
    
    transformer_primitive = []
    
    transformer_primitive.append(T.Resize(size=i_shape))
    if h_flip > 0:
      transformer_primitive.append(T.RandomHorizontalFlip(p=h_flip))
    if t_crop:
      transformer_primitive.append(T.RandomCrop(size=i_shape))
    transformer_primitive.append(T.ToTensor())
    transformer_primitive.append(T.Normalize(mean=normalization_mean, std=normalization_std))
    if rea:
      transformer_primitive.append(T.RandomErasing(p=0.5, scale=(0.02, 0.4), value = kwargs.get('rea_value', 0)))
    self.transformer = T.Compose(transformer_primitive)

  def setup(self,datacrawler, mode='train', batch_size=32, instance = 6, workers = 8):
    """ Setup the data generator.

    Args:
      workers (int): Number of workers to use during data retrieval/loading
      datacrawler (VeRiDataCrawler): A DataCrawler object that has crawled the data directory
      mode (str): One of 'train', 'test', 'query'. 
    """
    if datacrawler is None:
      raise ValueError("Must pass DataCrawler instance. Passed `None`")
    self.workers = workers * self.gpus

    # If training, get images whose labels are 0-97
    # If testing, get images whose labels are 98-196
    if mode == "train":
      self.__dataset = TDataSet(datacrawler.metadata["train"]["crawl"]+datacrawler.metadata["test"]["crawl"], self.transformer, range(0,98))
    if mode == "train-gzsl":
      self.__dataset = TDataSet(datacrawler.metadata["train"]["crawl"], self.transformer, range(0,98))
    elif mode == "zsl" or mode == "test":
      self.__dataset = TDataSet(datacrawler.metadata["train"]["crawl"] + datacrawler.metadata["test"]["crawl"], self.transformer, range(98,196))
    elif mode == "gzsl":  # handle generalized zero shot learning testing...
      self.__dataset = TDataSet(datacrawler.metadata["train"]["crawl"] + datacrawler.metadata["test"]["crawl"], self.transformer, range(0,196))
    else:
      raise NotImplementedError()
    
    if mode == "train" or mode == "train-gzsl":
      self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, \
                                        shuffle=True, \
                                        num_workers=self.workers, drop_last=True, collate_fn=self.collate_simple)
      self.num_entities = 98
    elif mode == "zsl" or mode == "test":
      self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, \
                                        shuffle = False, 
                                        num_workers=self.workers, drop_last=True, collate_fn=self.collate_simple)
      self.num_entities = 98
    elif mode == "gzsl":
      self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, \
                                        shuffle = False, 
                                        num_workers=self.workers, drop_last=True, collate_fn=self.collate_simple)
      self.num_entities = 196
    else:
      raise NotImplementedError()
  def collate_simple(self,batch):
    img, pid, _ = zip(*batch)
    pid = torch.tensor(pid, dtype=torch.int64)
    return torch.stack(img, dim=0), pid
  def collate_with_camera(self,batch):
    img, pid, idx = zip(*batch)
    pid = torch.tensor(pid, dtype=torch.int64)
    idx = torch.tensor(idx, dtype=torch.int64)
    return torch.stack(img, dim=0), pid, idx

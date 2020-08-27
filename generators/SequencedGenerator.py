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
  def __init__(self,dataset, transform):
    self.dataset = dataset
    self.transform = transform
  def __len__(self):
    return len(self.dataset)
  def __getitem__(self,idx):
    img, pid, cid = self.dataset[idx]
    img_arr = self.transform(self.load(img))
    return img_arr, pid, cid, img
  
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
class SequencedGenerator:
  def __init__(self,gpus, i_shape = (208,208), normalization_mean = 0.5, normalization_std = 0.5, normalization_scale = 1./255., h_flip = 0.5, t_crop = True, rea = True, **kwargs):
    """ Data generator for training and testing. Works with the VeriDataCrawler. Should work with any crawler working on VeRi-like data. Not yet tested with VehicleID. Only  use with VeRi.

    Generates batches of batch size CONFIG.TRANSFORMATION.BATCH_SIZE, with CONFIG.TRANSFORMATION.INSTANCE unique ids. So if BATCH_SIZE=36 and INSTANCE=6, then generate batch of 36 images, with 6 identities, 6 image per identity. See arguments of setup function for INSTANCE.

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

  def setup(self,datacrawler, mode='train', batch_size=32, instance = 8, workers = 8):
    """ Setup the data generator.

    Args:
      workers (int): Number of workers to use during data retrieval/loading
      datacrawler (VeRiDataCrawler): A DataCrawler object that has crawled the data directory
      mode (str): One of 'train', 'test', 'query'. 
    """
    if datacrawler is None:
      raise ValueError("Must pass DataCrawler instance. Passed `None`")
    self.workers = workers * self.gpus

    if mode == "train":
      self.__dataset = TDataSet(datacrawler.metadata[mode]["crawl"], self.transformer)
    elif mode == "test":
      # For testing, we combine images in the query and testing set to generate batches
      self.__dataset = TDataSet(datacrawler.metadata[mode]["crawl"] + datacrawler.metadata["query"]["crawl"], self.transformer)
    else:
      raise NotImplementedError()
    
    if mode == "train":
      self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, \
                                        sampler = TSampler(datacrawler.metadata[mode]["crawl"], batch_size=batch_size*self.gpus, instance=instance*self.gpus), \
                                        num_workers=self.workers, collate_fn=self.collate_simple)
      self.num_entities = datacrawler.metadata[mode]["pids"]
    elif mode == "test":
      self.dataloader = TorchDataLoader(self.__dataset, batch_size=batch_size*self.gpus, \
                                        shuffle = False, 
                                        num_workers=self.workers, collate_fn=self.collate_with_camera)
      self.num_entities = len(datacrawler.metadata["query"]["crawl"])
    else:
      raise NotImplementedError()
    
  def collate_simple(self,batch):
    img, pid, _, _ = zip(*batch)
    pid = torch.tensor(pid, dtype=torch.int64)
    return torch.stack(img, dim=0), pid
  def collate_with_camera(self,batch):
    img, pid, cid, path = zip(*batch)
    pid = torch.tensor(pid, dtype=torch.int64)
    cid = torch.tensor(cid, dtype=torch.int64)
    return torch.stack(img, dim=0), pid, cid, path

import torch
import torchvision
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


class ClassedGenerator:
    """ ClassedGenerator returns images only for the specified classes """
    def __init__(self, preload = "", gpus = 1, i_shape=(128,128), normalization_mean = 0.5, normalization_std = 0.5, normalization_scale = 1./255., h_flip = 0.0, t_crop = False, rea = False, **kwargs):
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
        #transformer_primitive.append(T.Normalize(mean=normalization_mean, std=normalization_std))
        if rea:
            transformer_primitive.append(T.RandomErasing(p=0.5, scale=(0.02, 0.4), value = kwargs.get('rea_value', 0)))
        self.transformer = T.Compose(transformer_primitive)

    def setup(self, datacrawler, mode = "train", batch_size=32, workers = 8, preload_classes = []):
        """ Setup the data generator.

        Args:
            workers (int): Number of workers to use during data retrieval/loading
            datacrawler (VeRiDataCrawler): A DataCrawler object that has crawled the data directory
            mode (str): One of 'train', 'test', 'query'. 
        """
        if datacrawler is None:
            raise ValueError("Must pass DataCrawler instance. Passed `None`")
        self.workers = workers * self.gpus

        train_mode = True if mode == "train" else False
        target_convert = None
        if datacrawler in ["MNIST", "CIFAR10", "CIFAR100"]:
            __dataset = getattr(torchvision.datasets, datacrawler)
            self.__dataset = __dataset(root="./"+datacrawler, train=train_mode, download=True, transform = self.transformer)
            # Add extra channel to MNIST
            if datacrawler == "MNIST":
                self.__dataset.data = self.__dataset.data.unsqueeze(3)
            # Need to handle issue where CIFAR10, CIFAR100 torch dataset downloaders load into list, instead of torch tensor
            if datacrawler in ["CIFAR10", "CIFAR100"]:
                if type(self.__dataset.targets).__name__ ==  "list":
                    target_convert = "list"
                    self.__dataset.targets = torch.Tensor(self.__dataset.targets).int()
                #if type(self.__dataset.data).__name__ == "ndarray":
                #    pdb.set_trace()
                #self.__dataset.data = torch.from_numpy(self.__dataset.data).double()

            if len(preload_classes) > 0:
                valid_idxs = self.__dataset.targets == preload_classes[0]
                for _remaining in preload_classes[1:]:
                    valid_idxs += self.__dataset.targets == _remaining
                self.__dataset.targets = self.__dataset.targets[valid_idxs]
                self.__dataset.data = self.__dataset.data[valid_idxs]
            if target_convert == "list":
                self.__dataset.targets = self.__dataset.targets.tolist()
        else:
            raise NotImplementedError()

        self.dataloader = TorchDataLoader(self.__dataset, batch_size = batch_size*self.gpus, shuffle=True, num_workers = self.workers)


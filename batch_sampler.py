import os
import collections
import numpy as np
from torch.utils.data import Sampler
from utils import pad_batch

np.random.seed(0)

class RandomBatchSampler(Sampler):
    """
    Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset) : dataset to sample from
    """
    def __init__(self,data_source,batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):

        size = len(self.data_source) - len(self.data_source) % self.batch_size
        data = np.array(list(range(size)))
        data = data.reshape(-1,self.batch_size)
        np.random.shuffle(data)
        data = data.flatten().tolist()
        if size < len(self.data_source):
            data.extend(list(range(size,len(self.data_source))))
        return iter(data)

    def __len__(self):
        return len(self.data_source)

    def __len__(self):
        return len(self.data_source)


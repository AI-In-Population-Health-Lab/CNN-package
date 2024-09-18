import pandas as pd
import torch
import sys
from collections import Counter
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    # data is a dataframe; parsing happens elsewhere since we want to split the target training data for validation
    # label_key is the non-dummied column key of the label
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return self.features.shape[0]


    def __getitem__(self, index):
        row = self.features.iloc[index]
        target = self.labels.iloc[index]
        return (torch.tensor(row, dtype=torch.float32), torch.tensor(target, dtype=torch.int64)) # CPU tensors


    def __Nfeatures__(self):
        return self.features.shape[1]


    def __Nlabels__(self):
        return len(Counter(self.labels).keys())

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    
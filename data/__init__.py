import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
from .datasets import dataset_folder


def get_dataset(opt):
    dset_train_lst = []
    dset_val_lst = []
    if opt.eval_mode !=True :
        if len(opt.classes) == 1:
            root = opt.dataroot + '/' + 'train'
            dset = dataset_folder(opt, root)
            total_samples = len(dset)
            train_size = 48000
            val_size = 6000

            train_dataset, val_dataset,_= random_split(dset, [train_size, val_size,total_samples-train_size-val_size])
            return train_dataset, val_dataset
        else:
            for cls in opt.classes:
                root = opt.dataroot + '/' + cls + '/train'
                dset = dataset_folder(opt, root)
                total_samples = len(dset)
                train_size = 48000
                val_size = 6000
                train_dataset, val_dataset,_= random_split(dset, [train_size, val_size,total_samples-train_size-val_size])
                dset_train_lst.append(train_dataset)
                dset_val_lst.append(val_dataset)
            return torch.utils.data.ConcatDataset(dset_train_lst),torch.utils.data.ConcatDataset(dset_val_lst)
    else :
        dset_val_lst = []
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_val_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_val_lst),None



def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    train_dataset, val_dataset= get_dataset(opt)
    train_sampler = get_bal_sampler(train_dataset) if opt.class_bal else None
    val_sampler = get_bal_sampler(val_dataset) if opt.class_bal else None
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=train_sampler,
                                              num_workers=int(opt.num_threads))
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=val_sampler,
                                              num_workers=int(opt.num_threads))
    
    return train_data_loader , val_data_loader

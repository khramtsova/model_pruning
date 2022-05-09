
import os

import pytorch_lightning as pl
# from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
from torch.utils.data import TensorDataset


import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
from torchvision import transforms, datasets


class CIFARDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        image_size = (32, 32)

        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(image_size),
            transforms.Normalize(data_mean, data_std),
        ])

    def setup(self, stage=None):

        self.dataset_train = datasets.CIFAR10(self.args.data_path, train=True,
                                              transform=self.train_transform, download=False)

        self.dataset_val = datasets.CIFAR10(self.args.data_path, train=False,
                                            transform=self.test_transform, download=False)

        self.dataset_test = datasets.CIFAR10(self.args.data_path, train=False,
                                            transform=self.test_transform, download=False)

        return

    def train_dataloader(self) -> DataLoader:
        # choose the training and test datasets
        train_loader = DataLoader(self.dataset_train,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  #drop_last=True
                                  )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.dataset_val,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.num_workers,
                                pin_memory=True)

        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.dataset_val,
                                batch_size=self.args.batch_size,
                                shuffle=False,
                                num_workers=self.args.num_workers,
                                pin_memory=True)
        return test_loader

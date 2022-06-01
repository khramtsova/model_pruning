
import pytorch_lightning as pl

from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
from torchvision import transforms, datasets


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.data_dir = data_dir  # where you put the data
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset_train = datasets.ImageFolder(self.args.data_path + '/train', transform=self.train_transform)
            self.dataset_val = datasets.ImageFolder(self.args.data_path + '/val', transform=self.test_transform)

            # train_s = int(len(im_full) * 0.9)
            # val_s = len(im_full) - train_s
            # self.train, self.val = random_split(im_full, [train_s, val_s])
        if stage == 'test' or stage is None:
            self.dataset_test = datasets.ImageFolder(self.args.data_path + '/val', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

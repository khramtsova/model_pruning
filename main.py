from os import listdir
from os.path import isfile, join

import gc
import pytorch_lightning as pl
# from argparse import ArgumentParser
import argparse
import wandb
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import SubsetRandomSampler, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from torchvision import transforms

import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics import functional as FM
from torchmetrics.functional import accuracy, precision

from pytorch_lightning.loggers import CSVLogger
import torchvision.models as models

from pytorch_lightning.loggers import WandbLogger
# from topo.topo_utils import get_diagrams_feature_vectors, wasserstein_d

# RSC-related libraries
from models.imagenet_resnet import resnet18, resnet50
from models.cifar_resnet import resnet20, resnet32, resnet56
from utils.utils import precisions

# from pruning.pruning_geometric_mean_mine import Mask
from pruning.pruning_geometric_mean_initial import Mask
from data.CIFAR_data_module import CIFARDataModule

# Topology-related libraries
# from topo_utils.topo_freeze import TopoWeightFreezer, get_conv_list


class Net(pl.LightningModule):

    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.save_hyperparameters()

        if args.arch == 'resnet18':
            net = resnet18(pretrained=False, classes=args.n_classes)
        elif args.arch == 'resnet50':
            net = resnet50(pretrained=False, classes=args.n_classes)
        else:
            net = resnet20(num_classes=args.n_classes)
        net = net.cuda()
        """
        self.mask = Mask(net, args)
        self.mask.init_length(net, args.rate_norm, args.rate_dist)
        self.mask.init_mask( args.dist_type, net)
        #    m.if_zero()
        """

        self.mask = Mask(net, args)
        self.mask.init_length()
        self.mask.model = net
        self.mask.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
        self.mask.do_mask()
        self.mask.do_similar_mask()
        self.net = self.mask.model
        #    m.if_zero()
        # net = self.mask.do_similar_mask(net)
        # self.topo_weight_freezer = TopoWeightFreezer(self.net, persentage_to_freeze=args.freeze)

        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):

        optimizer = optim.SGD(self.parameters(), weight_decay=.0005, momentum=self.args.momentum,
                              nesterov=self.args.nesterov, lr=self.args.learning_rate)
        # every 30 epochs - decrease by 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.schedule,
                                                   gamma=self.args.gammas)

        return [optimizer], [scheduler]

    def on_train_start(self):
        pl.seed_everything(self.args.rand_seed)

    def training_step(self, batch, batch_indx):
        input, target = batch
        # self.net = self.mask.do_similar_mask(self.net)
        output = self.net(input)
        # for i in
        loss = self.criterion(output, target)
        prec1, prec5 = precision(output, target, top_k=1, num_classes=self.args.n_classes),  \
                       precision(output, target, top_k=5,  num_classes=self.args.n_classes)
        acc = accuracy(output, target, num_classes=self.args.n_classes)
        self.log("train/loss", loss)
        self.log("train/prec_1", prec1)
        self.log("train/prec_5", prec5)
        self.log("train/acc", acc)
        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # My version:
        # self.net = self.mask.do_grad_mask(self.net)
        self.mask.do_grad_mask()

    def validation_step(self, batch, batch_indx):
        input, target = batch
        output = self.net(input)
        loss = self.criterion(output, target)
        prec1, prec5 = precision(output, target, top_k=1, num_classes=self.args.n_classes),  \
                       precision(output, target, top_k=5,  num_classes=self.args.n_classes)
        acc = accuracy(output, target, num_classes=self.args.n_classes)
        # acc = cls_pred.eq(class_l.data).sum().detach().cpu().item() / cls_pred.size(0)
        self.log("val/loss", loss)
        self.log("val/prec_1", prec1)
        self.log("val/prec_5", prec5)
        self.log("val/acc", acc)

    def on_train_epoch_end(self):
        self.mask.model = self.net
        self.mask.if_zero()
        self.mask.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
        self.mask.do_mask()
        self.mask.do_similar_mask()
        self.mask.if_zero()
        self.net = self.mask.model
        self.net = self.net.cuda()
        """
        #  My version
        if self.current_epoch > 5:
            # self.m.model = self.net
            self.mask.if_zero(self.net)
            self.net = self.mask.do_similar_mask(self.net)

            self.mask.init_mask(args.dist_type, self.net)
            # self.m.do_mask()
            self.net = self.mask.do_similar_mask(self.net)
            self.net = self.mask.do_reinit(self.net)
            self.mask.if_zero(self.net)
            #self.net = self.m.model
        """

    def test_step(self, batch, batch_indx):
        input, target = batch
        output = self.net(input)
        loss = self.criterion(output, target)
        prec1, prec5 = precision(output, target, top_k=1, num_classes=self.args.n_classes), \
                       precision(output, target, top_k=5, num_classes=self.args.n_classes)
        acc = accuracy(output, target, num_classes=self.args.n_classes)
        # acc = cls_pred.eq(class_l.data).sum().detach().cpu().item() / cls_pred.size(0)
        self.log("test/prec_1", prec1)
        self.log("test/prec_5", prec5)
        self.log("test/acc", acc)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--data_path', default='/opt/data/CIFAR10/', type=str, help='Path to dataset')
        parser.add_argument('--arch', metavar='ARCH', default='resnet20', help='model architecture')
        parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                            help='Choose between Cifar10/100 and ImageNet.')

        parser.add_argument('--n_classes', default=10, type=int, help='Number of classes')

        # Optimization options
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
        parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
        parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
        parser.add_argument('--nesterov', type=bool, default=False, help='Nesterov')

        parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gammas', type=float, nargs='+', default=0.2,
                            help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

        # compress rate
        # Note1: rate_norm = 0.9 means pruning 10% filters by norm-based criterion,
        # rate_dist = 0.3 means pruning 30% filters by distance-based criterion.
        parser.add_argument('--rate_norm', type=float, default=1, help='the remaining ratio of pruning based on Norm')
        parser.add_argument('--rate_dist', type=float, default=0.4,
                            help='the reducing ratio of pruning based on Geometric Median')

        parser.add_argument('--layer_begin', type=int, default=0, help='compress layer of model')
        parser.add_argument('--layer_end', type=int, default=54, help='compress layer of model')
        parser.add_argument('--layer_inter', type=int, default=3, help='compress layer of model')
        parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')

        parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true',
                            help='use state dcit or not')
        parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true',
                            help='use pre-trained model or not')
        parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')
        parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'],
                            help='distance type of GM')

        parser.add_argument("--num_workers", default=6, type=int, help="typically, the number of cpu")

        parser.add_argument("--rsc", action='store_true', help="if rsc is used")
        parser.add_argument("--topo_freeze", action='store_true', help="if topo_freeze is used")

        parser.add_argument("--log_dir", default='logs/DEL', help="Used by the logger to save logs")
        parser.add_argument("--freeze", default=0.3, type=float, help="freeze_percentage")
        parser.add_argument("--rand_seed", default=0, type=int, help="Random seed")
        parser.add_argument("--starting_epoch", default=1, type=int, help="Starting epoch to apply the freeze")
        return parser

    @staticmethod
    def add_program_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser(description='PACS RSC training')
    parser = Net.add_model_specific_args(parser)
    parser = Net.add_program_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data = CIFARDataModule(args)

    image_model = Net(args=args)
    """
    logger = TensorBoardLogger(project='DEL',
                               save_dir=args.log_dir,
                               job_type='baseline',
                               # entity='khramtsova',
                               )

    logger.watch(image_model,  log="all")
    """
    checkpoint_callback = ModelCheckpoint(# monitor="val_acc",
                                          save_top_k=-1,
                                          save_last=True,
                                          # period=10,
                                          every_n_epochs=1,
                                          # mode="max"
                                          )
    trainer = pl.Trainer.from_argparse_args(args,
                                            #checkpoint_callback=checkpoint_callback,
                                            callbacks=[checkpoint_callback],
                                            # callbacks=[lr_monitor],
                                            progress_bar_refresh_rate=300,
                                            #logger=logger,
                                            # num_sanity_val_steps=-1,
                                            num_nodes=1,
                                            # enable_progress_bar=False,
                                            # detect_anomaly=True
                                            )

    trainer.fit(image_model, data)
    trainer.test(model=image_model, ckpt_path='best', datamodule=data)

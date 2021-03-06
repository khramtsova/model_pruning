
import torch
import numpy as np
from scipy.spatial import distance
import random
from gph.python import ripser_parallel


class Mask:
    def __init__(self, model, hparams):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

        self.args = hparams
        self.filters_to_prune= {}

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            # print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm_np.argsort()[filter_pruned_num:]
            filter_small_index = norm_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
            # for x1, x2 in enumerate(filter_large_index):
            #     for y1, y2 in enumerate(filter_large_index):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_index).cuda()
            weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter index with largest similarity == small distance
            similar_large_index = similar_sum.argsort()[similar_pruned_num:]
            similar_small_index = similar_sum.argsort()[:  similar_pruned_num]

            similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

            #print('filter_large_index', filter_large_index)
            #print('filter_small_index', filter_small_index)
            #print('similar_sum', np.sort(similar_sum))
            #print('similar_large_index', similar_large_index)
            #print('similar_small_index', similar_small_index)
            # print('similar_index_for_filter', np.sort(similar_index_for_filter))
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_index_for_filter)):
                codebook[
                similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
            # print("similar index done")
        else:
            pass
        return codebook, similar_index_for_filter

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            self.distance_rate[index] = 1
        for key in range(self.args.layer_begin, self.args.layer_end + 1, self.args.layer_inter):
            self.compress_rate[key] = rate_norm_per_layer
            self.distance_rate[key] = rate_dist_per_layer
        # different setting for  different architecture
        if self.args.arch == 'resnet20':
            last_index = 57
        elif self.args.arch == 'resnet32':
            last_index = 93
        elif self.args.arch == 'resnet56':
            last_index = 165
        elif self.args.arch == 'resnet110':
            last_index = 327

        # ImageNet-related archs
        elif self.args.arch == 'resnet18':
            # last index include last fc layer
            last_index = 60
            skip_list = [21, 36, 51]
        elif self.args.arch == 'resnet34':
            last_index = 108
            skip_list = [27, 54, 93]
        elif self.args.arch == 'resnet50':
            last_index = 159
            skip_list = [12, 42, 81, 138]
        elif self.args.arch == 'resnet101':
            last_index = 312
            skip_list = [12, 42, 81, 291]
        elif self.args.arch == 'resnet152':
            last_index = 465
            skip_list = [12, 42, 117, 444]
        self.mask_index = [x for x in range(0, last_index, 3)]
        # skip downsample layer
        if self.args.skip_downsample == 1:
            for x in skip_list:
                self.compress_rate[x] = 1
                self.mask_index.remove(x)
                # print(self.mask_index)

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # mask for norm criterion
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                           self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                #if self.args.use_cuda:
                self.mat[index] = self.mat[index].cuda()

                # # get result about filter index
                # self.filter_small_index[index], self.filter_large_index[index] = \
                #     self.get_filter_index(item.data, self.compress_rate[index], self.model_length[index])

                # mask for distance criterion
                self.similar_matrix[index], filter_to_prune = self.get_filter_similar(item.data, self.compress_rate[index],
                                                                     self.distance_rate[index],
                                                                     self.model_length[index], dist_type=dist_type)
                if index in self.filters_to_prune.keys():
                    # check if the value has changed
                    if list(np.sort(self.filters_to_prune[index])) != list(np.sort(filter_to_prune)):
                        print("The value of the filter", index, " has changed!")
                        print(list(np.sort(self.filters_to_prune[index])))
                        print(list(np.sort(filter_to_prune)))
                        self.filters_to_prune[index] = filter_to_prune
                else:
                    print("New filters")
                    self.filters_to_prune[index] = filter_to_prune
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                #if self.args.use_cuda:
                self.similar_matrix[index] = self.similar_matrix[index].cuda()
        # print("mask Ready")

    def init_mask_random(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                weight_torch = item.data
                codebook = np.ones(self.model_length[index])

                rand_index_for_filter = list(np.arange(weight_torch.size()[0]))
                random.shuffle(rand_index_for_filter)
                similar_pruned_num = int(weight_torch.size()[0] * self.distance_rate[index])
                rand_index_for_filter = rand_index_for_filter[:similar_pruned_num]
                kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
                for x in range(0, len(rand_index_for_filter)):
                    codebook[
                    rand_index_for_filter[x] * kernel_length: (rand_index_for_filter[x] + 1) * kernel_length] = 0

                self.similar_matrix[index] = codebook
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                #if self.args.use_cuda:
                self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print(self.similar_matrix)

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        #print("mask Done")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        #print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                # b = a * self.mat[index]
                b = a * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # return net
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))

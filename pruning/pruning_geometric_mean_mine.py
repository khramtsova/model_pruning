
import torch
import numpy as np
from scipy.spatial import distance
from gph.python import ripser_parallel


class Mask:
    def __init__(self, model, hparams):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        # self.mat = {}

        # self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

        self.replace_matrix = {}
        self.model = model
        self.args = hparams

    def get_filter_topo(self, weight_torch, length,
                        drop_per_epoch=0.2,
                        dist_type="l2"):
        codebook = np.ones(length)
        filters_to_reinit = {}
        filters_to_drop = []

        if len(weight_torch.size()) == 4:

            num_filters = weight_torch.size()[0]
            # how many to prune
            similar_pruned_num = int(num_filters * self.args.rate_dist)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            zero_mask = torch.count_nonzero(weight_vec, dim=1) == 0
            how_many_are_already_0 = sum(zero_mask)
            print("Zeros aready are", how_many_are_already_0, "vs total", len(weight_vec))
            # For the filters that have been zeroed already - keep them 0
            already_zero_filter_indx = torch.argwhere(zero_mask).flatten().cpu().numpy()

            if how_many_are_already_0 < similar_pruned_num:
                # Search for new filters to prune;
                n_filters_to_drop = int(num_filters * drop_per_epoch)
                if n_filters_to_drop == 0:
                    n_filters_to_drop = 1
                print("Next batch of filters to prune; dropping ", n_filters_to_drop , "filters")
                original_indices = np.arange(len(weight_vec))

                weight_vec_excluding_empty = weight_vec[torch.argwhere(~zero_mask).flatten()]
                reduced_indices = original_indices[torch.argwhere(~zero_mask).flatten().cpu()]
                weight_vec_cpu = weight_vec_excluding_empty.cpu().numpy()

                similar_matrix = distance.cdist(weight_vec_cpu, weight_vec_cpu, 'euclidean')
                topo_res = ripser_parallel(similar_matrix, metric="precomputed",
                                           maxdim=0, n_threads=-1, return_generators=True)
                # bd_pairs = topo_res['dgms']
                generators = topo_res['gens']

                filters_to_drop, filters_to_reinit = cluster_based_topo(generators, weight_vec_cpu,
                                                                        n_filters_to_drop)
                # Map back to the original indices
                filters_to_drop = reduced_indices[filters_to_drop]
                for old_key in list(filters_to_reinit.keys()):
                    filters_to_reinit[reduced_indices[old_key]] = filters_to_reinit.pop(old_key)
            else:
                print("Dropping for this layer is done")
            # join the filters with already empty
            print("Already zero", already_zero_filter_indx, "Filters to drop", filters_to_drop)
            filters_to_drop = list(filters_to_drop) + list(already_zero_filter_indx)
            print("REsulting Filters to drop", filters_to_drop, len(filters_to_drop))
            # [# of kernels x 3 x 3]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filters_to_drop)):
                codebook[
                filters_to_drop[x] * kernel_length: (filters_to_drop[x] + 1) * kernel_length] = 0
            print("topo index done")
        else:
            pass
        return codebook, filters_to_reinit

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

        #for index, item in enumerate(model.parameters()):
       #     self.compress_rate[index] = 1
        #for key in range(self.args.layer_begin, self.args.layer_end + 1, self.args.layer_inter):
        #    self.compress_rate[key] = rate_norm_per_layer

        # different setting for  different architecture
        if self.args.arch == 'resnet20':
            last_index = 57
        elif self.args.arch == 'resnet32':
            last_index = 93
        elif self.args.arch == 'resnet56':
            last_index = 165
        elif self.args.arch == 'resnet110':
            last_index = 327
        # to jump the last fc layer
        self.mask_index = [x for x in range(0, last_index, 3)]


    def init_mask(self, dist_type, drop_per_epoch):

        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                #if index == 0:
                #    print("Before initialisation", item)
                # mask for distance criterion
                self.similar_matrix[index], self.replace_matrix[index] = self.get_filter_topo(item.data,
                                                                                              # self.compress_rate[index],
                                                                                              self.model_length[index],
                                                                                              drop_per_epoch=drop_per_epoch,
                                                                                              dist_type=dist_type
                                                                                              )
                if index < 5:
                    print("INDEX", self.similar_matrix[index])
                #self.similar_matrix[index] = self.get_filter_similar(item.data, self.compress_rate[index],
                #                                                      self.distance_rate[index],
                #                                                     self.model_length[index], dist_type=dist_type)
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                #if self.args.gpus is not None:
                self.similar_matrix[index] = self.similar_matrix[index].cuda()
        # print("mask Ready")

    def do_similar_mask(self, model):
        for index, item in enumerate(model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        return model

    def do_reinit(self, model):
        for index, item in enumerate(model.parameters()):
            if index in self.mask_index:
                b = item.data.view(self.model_size[index][0], -1)
                if len(self.replace_matrix[index].keys()) > 0:
                    for filter_indx in self.replace_matrix[index].keys():
                        b[filter_indx] = self.replace_matrix[index][filter_indx]
                item.data = b.view(self.model_size[index])
        return model

    def if_zero(self, model):
        for index, item in enumerate(model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))

    def do_grad_mask(self, model):
        for index, item in enumerate(model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                # b = a * (1 - self.mat[index])
                # b = a * self.mat[index]

                b = a * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        return model
            # print("grad zero Done")
        # print("grad zero Done")


def cluster_based_topo(generators, weight_vec, n_filters_to_drop):

    # list of sets containing clusters
    cluster = []
    for el in generators[0]:
        stopping_criteria = sum([len(i) - 1 for i in cluster]) == n_filters_to_drop
        if stopping_criteria:
            break
        if len(cluster) > 0:
            cl_indx = [i for i in range(len(cluster)) if (el[1] in cluster[i] or el[2] in cluster[i])]
            if len(cl_indx) > 2:
                print("Error")
                raise ()
            if len(cl_indx) == 0:
                cluster.append({el[1], el[2]})

            # add to an existing cluster
            if len(cl_indx) == 1:
                cluster[cl_indx[0]].add(el[1])
                cluster[cl_indx[0]].add(el[2])

            # merge clusters
            if len(cl_indx) == 2:
                if cl_indx[1] > cl_indx[0]:
                    cluster_1, cluster_2 = cluster.pop(cl_indx[1]), cluster.pop(cl_indx[0])
                else:
                    cluster_1, cluster_2 = cluster.pop(cl_indx[0]), cluster.pop(cl_indx[1])
                cluster_1 |= cluster_2
                cluster.append(cluster_1)

        else:
            cluster.append({el[1], el[2]})

    # Go through the clusters and remove the most centered filter from the list to drop
    filters_to_reinit = {}
    for i in range(len(cluster)):
        sub_indexes = list(cluster[i])
        subvectors = weight_vec[sub_indexes]
        local_similarity_matrix = distance.cdist(subvectors, subvectors, 'euclidean')
        local_similar_sum = np.sum(np.abs(local_similarity_matrix), axis=0)
        # The most central filter
        similar_small_index = local_similar_sum.argsort()[0]
        # print(cluster[i])
        filters_to_reinit[sub_indexes[similar_small_index]] = torch.from_numpy(np.mean(subvectors, axis=0)).cuda()
        # Remove the most central filter from the list of filters to drop
        cluster[i].remove(sub_indexes[similar_small_index])

    filters_to_drop = list({x for _list in cluster for x in _list})

    return filters_to_drop, filters_to_reinit

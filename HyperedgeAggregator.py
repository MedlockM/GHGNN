import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import time

class HyperedgeMeanAggregator(nn.Module):
    """
    Aggregates a hyperedge's embeddings using mean of internal nodes' embeddings
    """

    def __init__(self, features_mat, hyp2nodes):
        """
        Initializes the aggregator for a specific hypergraph.
        features_mat -- function mapping LongTensor of node ids to FloatTensor of feature values.
        hyp2nodes -- dictionnary (key: hyperedge name, value: list of internal nodes name)
        """

        super(HyperedgeMeanAggregator, self).__init__()

        self.features_mat = features_mat
        self.hyp2nodes = hyp2nodes

    def compute(self, hyperedges):
        """
        hyperedges --- list of the hyperedges to compute the internal nodes aggregation
        """
        if type(hyperedges)==torch.Tensor:
            hyperedges=hyperedges.tolist()
        #print('hyperedges to compute in hyp agg', hyperedges)
        #print('hyperedges size', len(hyperedges))
        #print('hyp2nodes', self.hyp2nodes)
        #print('nodes/hyperedges :', nodes)
        #print("adj_dict =", self.adj_dict)
        internal_nodes_lists = [self.hyp2nodes[h] for h in hyperedges] # filtering of hyp2nodes
        #print('neighbors lists', neighbors_lists)
        to_internal_nodes = {k: v for (k,v) in zip(hyperedges, [self.hyp2nodes[h] for h in hyperedges])}
        #print('to_neighs', to_neighs)
        flatten_int_nodes_list = [i for item in internal_nodes_lists for i in item]
        unique_int_nodes_list = list(dict.fromkeys(flatten_int_nodes_list))
        #print('unique nodes list', unique_int_nodes_list)
        unique_int_nodes_dict = {n: i for i, n in enumerate(unique_int_nodes_list)}
        #print("nodes dict", unique_nodes_dict)
        mask = Variable(torch.zeros(len(hyperedges), len(unique_int_nodes_list)))
        column_indices = [unique_int_nodes_dict[n] for to_int_nodes in to_internal_nodes.values() for n in to_int_nodes]
        #print('column indices', column_indices)
        row_indices = [i for i in range(len(internal_nodes_lists)) for j in range(len(internal_nodes_lists[i]))]
        #print('row indices', row_indices)
        mask[row_indices, column_indices] = 1
        num_int_nodes = mask.sum(1, keepdim=True)
        mask = mask.div(num_int_nodes)
        #print('mask', mask)
        #print('Long Tensor', torch.LongTensor([int(i) for i in list(unique_nodes_list)]))
        #print('features mat', self.features_mat)
        #print('fn activation in agg')
        if type(self.features_mat)==nn.Embedding:
            embed_matrix = self.features_mat(torch.LongTensor([int(i) for i in list(unique_int_nodes_list)]))
        else:
            embed_matrix = self.features_mat([int(i) for i in list(unique_int_nodes_list)])
        #print('internal nodes embed matrix size', embed_matrix.size())
        agg_feats = mask.mm(embed_matrix)
        #print('aggregated hyperedges features size:', agg_feats.size())
        #print('end node agg')
        return agg_feats

class HyperedgeMaxAggregator(nn.Module):
    """
    Aggregates a hyperedge's embeddings using max of its internal nodes embeddings
    """

    def __init__(self, features_mat, hyp2nodes):
        """
        Initializes the aggregator for a specific hypergraph.
        features_mat -- function mapping LongTensor of node ids to FloatTensor of feature values.
        hyp2nodes -- dictionnary (key: hyperedge name, value: list of internal nodes name)
        """

        super(HyperedgeMaxAggregator, self).__init__()

        self.features_mat = features_mat
        self.hyp2nodes = hyp2nodes

    def compute(self, hyperedges):
        """
        hyperedges --- list of hyperedges to compute the internal nodes aggregation
        """
        #print("adj_dict =", self.adj_dict)
        #print('nodes in agg', nodes)
        if type(hyperedges)==torch.Tensor:
            hyperedges=hyperedges.tolist()
        #print('hyperedges in agg:', hyperedges)
        for i, n in enumerate(hyperedges):
            #print('i in hyp agg', i)
            #print('n in hyp agg', n)
            list_internal_nodes = [int(j) for j in self.hyp2nodes[n]] # for each node, get his neighbors
            #print('list int nodes :', list_internal_nodes)
            #print('features', self.features_mat)
            int_nodes_mat = self.features_mat(torch.LongTensor(list_internal_nodes)) # get internal nodes features
            #print('int nodes mat :', int_nodes_mat.size())
            #print('neighbors features', neighbors_mat)
            max_pool, max_indices = torch.max(int_nodes_mat, 0) # extract max value according to each feature
            #print('ha max pool size', max_pool.size())
            if i == 0:
                #print('begin')
                embed_matrix = max_pool[None, :]
            else :
                embed_matrix = torch.cat([embed_matrix, max_pool[None, :]], 0)
            #print('ha embed mat size', embed_matrix.size())
        #print(embed_matrix)
        return embed_matrix

    def compute2(self, hyperedges):
        """
        hyperedges --- list of hyperedges to compute the internal nodes aggregation
        """
        #print("adj_dict =", self.adj_dict)
        #print('nodes in agg', nodes)
        if type(hyperedges)==torch.Tensor:
            hyperedges=hyperedges.tolist()
        #print('hyperedges in agg:', hyperedges)
        internal_nodes_lists = [self.hyp2nodes[h] for h in hyperedges]  # filtering of hyp2nodes
        # print('neighbors lists', neighbors_lists)
        to_internal_nodes = {k: v for (k, v) in zip(hyperedges, [self.hyp2nodes[h] for h in hyperedges])}
        # print('to_neighs', to_neighs)
        flatten_int_nodes_list = [i for item in internal_nodes_lists for i in item]
        unique_int_nodes_list = list(dict.fromkeys(flatten_int_nodes_list))
        # print('unique nodes list', unique_int_nodes_list)
        unique_int_nodes_dict = {n: i for i, n in enumerate(unique_int_nodes_list)}
        # print("nodes dict", unique_nodes_dict)
        mask = Variable(torch.zeros(len(hyperedges), len(unique_int_nodes_list)))
        column_indices = [unique_int_nodes_dict[n] for to_int_nodes in to_internal_nodes.values() for n in to_int_nodes]
        # print('column indices', column_indices)
        row_indices = [i for i in range(len(internal_nodes_lists)) for j in range(len(internal_nodes_lists[i]))]
        # print('row indices', row_indices)
        mask[row_indices, column_indices] = 1
        #print('mask', mask.size())
        mask = mask.unsqueeze(-1)
        #print('mask', mask.size())

        if type(self.features_mat)==nn.Embedding:
            feat = self.features_mat(torch.LongTensor([int(i) for i in list(unique_int_nodes_list)]))
        else:
            feat = self.features_mat([int(i) for i in list(unique_int_nodes_list)])

        #print('feat size', feat.size())
        mask = mask.repeat(1, 1, feat.size()[1])

        #print('mask', mask.size())

        #print('feat', feat.size())
        feat = feat.unsqueeze(-1)
        #print('feat', feat.size())
        feat = feat.repeat(len(hyperedges), 1, 1)
        #print('feat', feat.size())
        mask_dim1, mask_dim2, mask_dim3 = mask.size()[0], mask.size()[1], mask.size()[2]
        feat = torch.reshape(feat, (mask_dim1, mask_dim2, mask_dim3))
        #print('feat', feat.size())
        mul = torch.mul(feat, mask)

        #print('mul', mul.size())
        embed_matrix, i = torch.max(mul, 1)

        #print(embed_matrix)

        return embed_matrix

class HyperedgeAttAggregator(nn.Module):
    """
    Aggregates a hyperedge's embeddings using weighted average (attention module) of its internal nodes embeddings
    """

    def __init__(self, features_mat, hyp2nodes, in_features, embed_dim, hyperedges_embed, dropout=0.6, alpha=0.2, concat=True, base_model=None):
        super(HyperedgeAttAggregator, self).__init__()
        self.features_mat = features_mat # function to compute node embedding at previous layer
        self.hyp2nodes = hyp2nodes
        self.hyperedges_embed = hyperedges_embed # function to compute hyperedge embedding at previous layer
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.embed_dim = embed_dim  #
        #print(self.embed_dim)
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # concat = True for all layers except the output layer.
        if base_model!=None:
            self.base_model = base_model
        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.query_weight = nn.Parameter(torch.zeros(size=(in_features, embed_dim)))
        nn.init.xavier_uniform_(self.query_weight.data, gain=1.414)
        self.key_weight = nn.Parameter(torch.zeros(size=(in_features, embed_dim)))
        nn.init.xavier_uniform_(self.key_weight.data, gain=1.414)
        self.value_weight = nn.Parameter(torch.zeros(size=(in_features, embed_dim)))
        nn.init.xavier_uniform_(self.value_weight.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(embed_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def embed_n(self, nodes):

        if type(nodes) == torch.Tensor:
            nodes = nodes.tolist()
        if type(self.features_mat)==nn.Embedding:
            embed_matrix = self.features_mat(torch.LongTensor([int(i) for i in list(nodes)]))
        else:
            embed_matrix = self.features_mat([int(i) for i in list(nodes)])
        embed_matrix = torch.mm(embed_matrix, self.query_weight)
        return embed_matrix

    def embed_h(self, hyperedges):

        if type(hyperedges) == torch.Tensor:
            hyperedges = hyperedges.tolist()
        if type(self.hyperedges_embed)==nn.Embedding:
            embed_matrix = self.hyperedges_embed(torch.LongTensor([int(i) for i in list(hyperedges)]))
        else:
            embed_matrix = self.hyperedges_embed([int(i) for i in list(hyperedges)])
        embed_matrix = torch.mm(embed_matrix, self.query_weight)
        return embed_matrix

    def compute(self, hyperedges):
        if type(hyperedges) == torch.Tensor:
            hyperedges = hyperedges.tolist()

        internal_nodes_lists = [self.hyp2nodes[h] for h in hyperedges]  # filtering of hyp2nodes
        # print('neighbors lists', neighbors_lists)
        to_internal_nodes = {k: v for (k, v) in zip(hyperedges, [self.hyp2nodes[h] for h in hyperedges])}
        # print('to_neighs', to_neighs)
        flatten_int_nodes_list = [i for item in internal_nodes_lists for i in item]
        unique_int_nodes_list = list(dict.fromkeys(flatten_int_nodes_list))
        # print('unique nodes list', unique_int_nodes_list)
        unique_int_nodes_dict = {n: i for i, n in enumerate(unique_int_nodes_list)}

        if type(self.features_mat)==nn.Embedding:
            #print("feat size", self.features_mat.weight.size())
            nodes_embed = self.features_mat(torch.LongTensor([int(i) for i in list(unique_int_nodes_list)]))
        else:
            nodes_embed = self.features_mat([int(i) for i in list(unique_int_nodes_list)])
        #print("nodes embed", nodes_embed.size())

        if type(self.hyperedges_embed)==nn.Embedding:
            hyperedges_embed = self.hyperedges_embed(torch.LongTensor([int(i) for i in list(hyperedges)]))
        else:
            hyperedges_embed = self.hyperedges_embed([int(i) for i in list(hyperedges)])
        #print("hyperedges embed", nodes_embed.size())

        # print("nodes dict", unique_nodes_dict)
        mask = Variable(torch.ones(len(hyperedges), len(unique_int_nodes_list)))
        mask = torch.mul(mask, torch.finfo(torch.float64).min)
        #print("mask", mask.size())
        column_indices = [unique_int_nodes_dict[n] for to_int_nodes in to_internal_nodes.values() for n in to_int_nodes]
        # print('column indices', column_indices)
        row_indices = [i for i in range(len(internal_nodes_lists)) for j in range(len(internal_nodes_lists[i]))]
        # print('row indices', row_indices)

        #hyperedges_embed_query = torch.mm(hyperedges_embed, self.query_weight)
        nodes_embed_keys = torch.mm(nodes_embed, self.query_weight)
        nodes_embed_values = torch.mm(nodes_embed, self.value_weight)
        #print("nodes embed keys", nodes_embed_keys.size())
        #print("hyperedges embed query", hyperedges_embed_query.size())
        #print("nodes embed values", nodes_embed_values.size())
        scores = torch.mm(nodes_embed_keys, self.a)
        list_scores = scores.tolist()
        #print(list_scores)
        scores = np.repeat(list_scores, len(hyperedges), axis=1).transpose()
        #print('scores', scores)
        scores = torch.Tensor(scores)
        # Masked Attention
        mask[row_indices, column_indices] = scores[row_indices, column_indices]
        #print('mask', mask)
        mask = F.softmax(mask, dim=1)
        #print('mask', mask)
        #mask = F.dropout(mask, self.dropout, training=self.training)
        hyperedges_embed_prime = torch.matmul(mask, nodes_embed_keys)

        #print('(att) hyp prime size', hyperedges_embed_prime.size())

        return hyperedges_embed_prime

    def compute2(self, hyperedges):
        if type(hyperedges) == torch.Tensor:
            hyperedges = hyperedges.tolist()

        internal_nodes_lists = [self.hyp2nodes[h] for h in hyperedges]  # filtering of hyp2nodes
        # print('neighbors lists', neighbors_lists)
        to_internal_nodes = {k: v for (k, v) in zip(hyperedges, [self.hyp2nodes[h] for h in hyperedges])}
        # print('to_neighs', to_neighs)
        flatten_int_nodes_list = [i for item in internal_nodes_lists for i in item]
        unique_int_nodes_list = list(dict.fromkeys(flatten_int_nodes_list))
        # print('unique nodes list', unique_int_nodes_list)
        unique_int_nodes_dict = {n: i for i, n in enumerate(unique_int_nodes_list)}

        if type(self.features_mat)==nn.Embedding:
            #print("feat size", self.features_mat.weight.size())
            nodes_embed = self.features_mat(torch.LongTensor([int(i) for i in list(unique_int_nodes_list)]))
        else:
            nodes_embed = self.features_mat([int(i) for i in list(unique_int_nodes_list)])
        #print("nodes embed", nodes_embed.size())

        if type(self.hyperedges_embed)==nn.Embedding:
            hyperedges_embed = self.hyperedges_embed(torch.LongTensor([int(i) for i in list(hyperedges)]))
        else:
            hyperedges_embed = self.hyperedges_embed([int(i) for i in list(hyperedges)])
        #print("hyperedges embed", nodes_embed.size())

        # print("nodes dict", unique_nodes_dict)
        mask = Variable(torch.ones(len(hyperedges), len(unique_int_nodes_list)))
        mask = torch.mul(mask, torch.finfo(torch.float64).min)
        #print("mask", mask.size())
        column_indices = [unique_int_nodes_dict[n] for to_int_nodes in to_internal_nodes.values() for n in to_int_nodes]
        # print('column indices', column_indices)
        row_indices = [i for i in range(len(internal_nodes_lists)) for j in range(len(internal_nodes_lists[i]))]
        # print('row indices', row_indices)

        hyperedges_embed_query = torch.mm(hyperedges_embed, self.query_weight)
        nodes_embed_keys = torch.mm(nodes_embed, self.query_weight)
        nodes_embed_values = torch.mm(nodes_embed, self.value_weight)
        #print("nodes embed keys", nodes_embed_keys.size())
        #print("hyperedges embed query", hyperedges_embed_query.size())
        #print("nodes embed values", nodes_embed_values.size())
        scores = torch.mm(hyperedges_embed_query, nodes_embed_keys.t()) / math.sqrt(self.embed_dim)
        #print("scores size", scores.size())
        #print('scores', scores)

        # Masked Attention
        mask[row_indices, column_indices] = scores[row_indices, column_indices]
        #print('mask', mask)
        mask = F.softmax(mask, dim=1)
        #print('mask', mask)
        #mask = F.dropout(mask, self.dropout, training=self.training)
        hyperedges_embed_prime = torch.matmul(mask, nodes_embed_keys)

        #print('(att) hyp prime size', hyperedges_embed_prime.size())

        return hyperedges_embed_prime

#print(torch.finfo(torch.float64).min)

# hyp2nodes = {}
# hyp2nodes["0"]=["0","1","2"]
# hyp2nodes["1"]=["0","3"]
# hyp2nodes["2"]=["1"]
#
# hyperedges_matrix = [[1, 1], [1,1], [1, 1]]
# hyperedges_mat = nn.Embedding(3, 2)
# hyperedges_mat.weight = nn.Parameter(torch.FloatTensor(hyperedges_matrix), requires_grad=False)
#
# nodes = ["0", "1", "2", "3", "4"]
# node2hyperedges = {}
# for j in nodes:
#     node2hyperedges[j] = [i for i in hyp2nodes.keys() if j in hyp2nodes[i]]
#
#
# features_matrix = [[2, 4], [6, 9], [6, 6], [3, 3], [3, 3]]
# features_mat = nn.Embedding(5, 2)
# features_mat.weight = nn.Parameter(torch.FloatTensor(features_matrix), requires_grad=False)
#
# adj_dict = {"0": ["1", "2", "3"], "1": ["0", "2"], "2": ["0", "1"], "3": ["0"], "4": []}
# MA = HyperedgeAttAggregator(features_mat, hyp2nodes, 2, 5, hyperedges_mat)
# MA.compute(["0", "1", "2"])

# features_matrix = [[3, 4], [1, 9], [1, 6], [3, 3], [3, 3]]
# features_mat = nn.Embedding(5, 2)
# features_mat.weight = nn.Parameter(torch.FloatTensor(features_matrix), requires_grad=False)
# MA = HyperedgeMaxAggregator(features_mat, {"0" : ["1", "2", "3"], "1": ["0", "2"], "2": ["0", "1"], "3": ["0"], "4": []})
# start_time = time.time()
# MA.compute2(["0", "1", "2", "3"])
# end_time = time.time()
# time = end_time - start_time

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of its hyperedges embeddings
    """

    def __init__(self, hyperedges_embedding, node2hyperedges):
        """
        Initializes the hyperedge aggregator for a specific hypergraph.
        hyperedges_embedding -- function mapping LongTensor of hyperedges ids to FloatTensor of embedding values.
        node2hyperedges -- dictionnary (key: node name, value: list of hyperedges containing the node)
        """

        super(MeanAggregator, self).__init__()

        self.hyperedges_embedding = hyperedges_embedding
        self.node2hyperedges = node2hyperedges

    def compute(self, nodes):
        """
        Compute the hyperedges aggregation for each of the node in nodes
        nodes --- list of batch nodes

        """
        if type(nodes)==torch.Tensor:
            nodes=nodes.tolist()
        #print('nodes to compute in node mean agg', nodes)
        #print('nodes size', len(nodes))
        print("node2hyperedges =", self.node2hyperedges)

        hyperedges_lists = [self.node2hyperedges[node] for node in nodes]
        #print('neighbors lists', neighbors_lists)
        to_hyps = {k: v for (k,v) in zip(nodes, [self.node2hyperedges[node] for node in nodes])} # filtering of node2hyperedges
        #print('to_neighs', to_neighs)
        flatten_hyperedges_list = [i for item in hyperedges_lists for i in item]
        unique_hyperedges_list = list(dict.fromkeys(flatten_hyperedges_list))
        #print('unique hyperedges list', unique_hyperedges_list)
        unique_hyperedges_dict = {n: i for i, n in enumerate(unique_hyperedges_list)}
        #print("nodes dict", unique_nodes_dict)
        mask = Variable(torch.zeros(len(nodes), len(unique_hyperedges_list)))
        column_indices = [unique_hyperedges_dict[n] for to_hyp in to_hyps.values() for n in to_hyp]
        #print('column indices', column_indices)
        row_indices = [i for i in range(len(hyperedges_lists)) for j in range(len(hyperedges_lists[i]))]
        #print('row indices', row_indices)
        mask[row_indices, column_indices] = 1
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        #print('mask', mask)
        #print('Long Tensor', torch.LongTensor([int(i) for i in list(unique_nodes_list)]))
        #print('hyperedge embeddings', self.hyperedges_embedding)
        if type(self.hyperedges_embedding) == nn.Embedding:
            embed_matrix = self.hyperedges_embedding(torch.LongTensor([int(i) for i in list(unique_hyperedges_list)]))
        else:
            embed_matrix = self.hyperedges_embedding(list(unique_hyperedges_list))
        #print('feat hyperedges size in hyp agg', embed_matrix.size())
        agg_feats = mask.mm(embed_matrix)
        #print('aggregated nodes embed size:', agg_feats.size())
        #print('end hyp agg')
        return agg_feats

class MaxAggregator(nn.Module):
    """
    Aggregates a node's embeddings using max of its hyperedges embeddings
    """

    def __init__(self, hyperedges_embedding, node2hyperedges):
        """
        Initializes the aggregator for a specific hypergraph.
        hyperedges_embedding -- function mapping LongTensor of hyperedges ids to FloatTensor of embedding values.
        node2hyperedges -- dictionnary (key: node name, value: list of hyperedges containing the node)
        """

        super(MaxAggregator, self).__init__()

        self.hyperedges_embedding = hyperedges_embedding
        self.node2hyperedges = node2hyperedges

    def compute(self, nodes):
        """
        nodes --- list of nodes to compute the internal nodes aggregation
        """
        #print("adj_dict =", self.adj_dict)

        if type(nodes)==torch.Tensor:
            nodes=nodes.tolist()
            #print('converted nodes :', nodes)
        #print('nodes in agg', nodes)
        for i, n in enumerate(nodes):
            #print(i)
            #print(n)
            #print('n in agg', n)
            list_hyperedges = [int(j) for j in self.node2hyperedges[n]] # for each node, get his neighbors
            #print('list linked hyperedges :', list_hyperedges)
            #print('hyper feat full', self.hyperedges_embedding.weight.size())

            hyperedges_mat = self.hyperedges_embedding(torch.LongTensor(list_hyperedges)) # get internal nodes features
            #print('hyperedges feat sub :', hyperedges_mat.size())
            #print('neighbors features', neighbors_mat)
            max_pool, max_indices = torch.max(hyperedges_mat, 0) # extract max value according to each feature
            #print('max pool size', max_pool.size())
            if i == 0:
                #print('begin')
                embed_matrix = max_pool[None, :]
            else :
                #print('next')
                embed_matrix = torch.cat([embed_matrix, max_pool[None, :]], 0)
            #print('embed mat size', embed_matrix.size())
        return embed_matrix

    def compute2(self, nodes):
        """
        nodes --- list of nodes to compute the internal nodes aggregation
        """
        #print("adj_dict =", self.adj_dict)
        #print('nodes in agg', nodes)
        if type(nodes)==torch.Tensor:
            nodes=nodes.tolist()
        #print('nodes in agg:', nodes)
        internal_nodes_lists = [self.node2hyperedges[h] for h in nodes]  # filtering of node2hyperedges
        # print('neighbors lists', neighbors_lists)
        to_internal_nodes = {k: v for (k, v) in zip(nodes, [self.node2hyperedges[h] for h in nodes])}
        # print('to_neighs', to_neighs)
        flatten_int_nodes_list = [i for item in internal_nodes_lists for i in item]
        unique_int_nodes_list = list(dict.fromkeys(flatten_int_nodes_list))
        # print('unique nodes list', unique_int_nodes_list)
        unique_int_nodes_dict = {n: i for i, n in enumerate(unique_int_nodes_list)}
        # print("nodes dict", unique_nodes_dict)
        mask = Variable(torch.zeros(len(nodes), len(unique_int_nodes_list)))
        column_indices = [unique_int_nodes_dict[n] for to_int_nodes in to_internal_nodes.values() for n in to_int_nodes]
        # print('column indices', column_indices)
        row_indices = [i for i in range(len(internal_nodes_lists)) for j in range(len(internal_nodes_lists[i]))]
        # print('row indices', row_indices)
        mask[row_indices, column_indices] = 1
        #print('mask', mask.size())
        mask = mask.unsqueeze(-1)
        #print('mask', mask.size())

        if type(self.hyperedges_embedding)==nn.Embedding:
            feat = self.hyperedges_embedding(torch.LongTensor([int(i) for i in list(unique_int_nodes_list)]))
        else:
            feat = self.hyperedges_embedding([int(i) for i in list(unique_int_nodes_list)])

        #print('feat size', feat.size())
        mask = mask.repeat(1, 1, feat.size()[1])

        #print('mask', mask.size())

        #print('feat', feat.size())
        feat = feat.unsqueeze(-1)
        #print('feat', feat.size())
        feat = feat.repeat(len(nodes), 1, 1)
        #print('feat', feat.size())
        mask_dim1, mask_dim2, mask_dim3 = mask.size()[0], mask.size()[1], mask.size()[2]
        feat = torch.reshape(feat, (mask_dim1, mask_dim2, mask_dim3))
        #print('feat', feat.size())
        mul = torch.mul(feat, mask)

        #print('mul', mul.size())
        embed_matrix, i = torch.max(mul, 1)

        #print(embed_matrix)

        return embed_matrix

class NodeAttAggregator(nn.Module):
    """
        Simple PyTorch Implementation of the Graph Attention layer.
        """

    def __init__(self, features_mat, hyp2nodes, in_features, out_features, hyperedges_embed, dropout=0.6, alpha=0.2, concat=True):
        super(NodeAttAggregator, self).__init__()
        self.features_mat = features_mat # function to compute node embedding at previous layer
        self.hyp2nodes = hyp2nodes
        self.hyperedges_embed = hyperedges_embed # function to compute hyperedge embedding at previous layer
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # concat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.query_weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.query_weight.data, gain=1.414)
        self.key_weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.key_weight.data, gain=1.414)
        self.value_weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.value_weight.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(in_features, 1)))
        nn.init.xavier_uniform_(self.query_weight.data, gain=1.414)

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
        scores = torch.mm(hyperedges_embed_query, nodes_embed_keys.t()) / math.sqrt(self.out_features)
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
        return hyperedges_embed_prime.t()

'''hyp2nodes = {}
hyp2nodes["0"]=["0","1","2"]
hyp2nodes["1"]=["0","3"]


nodes = ["0", "1", "2", "3", "4"]
node2hyperedges = {}
for j in nodes:
    node2hyperedges[j] = [i for i in hyp2nodes.keys() if j in hyp2nodes[i]]


features_matrix = [[2, 4], [6, 9], [6, 6], [3, 3], [3, 3]]
features_mat = nn.Embedding(5, 2)
features_mat.weight = nn.Parameter(torch.FloatTensor(features_matrix), requires_grad=False)

adj_dict = {"0": ["1", "2", "3"], "1": ["0", "2"], "2": ["0", "1"], "3": ["0"], "4": []}


node_agg = MeanAggregator(features_mat, hyp2nodes)

hypere_features = nn.Embedding(len(hyp2nodes), 2)
hypere_features.weight = nn.Parameter(torch.zeros(len(hyp2nodes), 2))

hyperedge_enc = HyperedgeEncoder(2, 3, node_agg, hypere_features)

hyperedge_agg = HyperedgeMeanAggregator(lambda hyperedge : hyperedge_enc.compute(hyperedge), node2hyperedges)

print(hyperedge_agg.compute(nodes))'''


#features_matrix = [[2, 4], [6, 9], [6, 6], [3, 3], [3, 3]]
#features_mat = nn.Embedding(5, 2)
#features_mat.weight = nn.Parameter(torch.FloatTensor(features_matrix), requires_grad=False)
#MA = MaxAggregator(features_mat, {"0" : ["1", "2", "3"], "1": ["0", "2"], "2": ["0", "1"], "3": ["0"], "4": []})
#MA.forward(["3", "2"])
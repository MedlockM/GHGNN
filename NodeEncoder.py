import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    """
    Encodes a node's using 'convolutional' spatial approach
    """

    def __init__(self, feature_dim, embed_dim, node_aggregator, previous_nodes_embeddings, base_model=None):
        super(NodeEncoder, self).__init__()

        self.feat_dim = feature_dim
        self.node_aggregator = node_aggregator
        if base_model != None:
            self.base_model = base_model
            #print("model", self.base_model)
        self.embed_dim = embed_dim
        self.previous_nodes_embeddings = previous_nodes_embeddings
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)


    def compute(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        #print('node encoding')
        #print('nodes size', len(nodes))
        agg_feats = self.node_aggregator.compute(nodes)
        #print("node agg feat size", agg_feats.size())
        if type(self.previous_nodes_embeddings) == nn.Embedding:
            self_feats = torch.Tensor(self.previous_nodes_embeddings(torch.LongTensor([int(i) for i in nodes])))
        else:
            self_feats = torch.Tensor(self.previous_nodes_embeddings([int(i) for i in nodes]))

        #print("nodes self features size", self_feats.size())
        combined = torch.cat([self_feats, agg_feats], dim=1)
        #print('combined before embedding size', combined.size())
        combined = F.relu(self.weight.mm(combined.t()))
        #print('node combined after embedding size', combined.t().size())
        #print('node combined after embedding', combined)
        #print('end nodes encoding-------------')
        return combined.t()


#features_matrix = [[2, 4], [6, 9], [6, 6], [3, 3], [3, 3]]
#features_mat = nn.Embedding(5, 2)
#features_mat.weight = nn.Parameter(torch.FloatTensor(features_matrix), requires_grad=False)
#MA = MeanAggregator(features_mat, {"1" : ["2", "3", "4"], "2": ["1", "3"], "3": ["1", "2"], "4": ["1"], "5": []})

#nodes = ["3", "2"]

#enc = Encoder(2, 3, MA)

#enc.forward(nodes)

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

hyperedge_enc = HyperedgeEncoder(2, 2, node_agg, hypere_features)

hyperedge_agg = HyperedgeMeanAggregator(lambda hyperedge : hyperedge_enc.compute(hyperedge), node2hyperedges)

node_enc = NodeEncoder(2, 5, hyperedge_agg, features_mat)

print(node_enc.compute(nodes))'''


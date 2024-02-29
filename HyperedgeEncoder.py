import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class HyperedgeEncoder(nn.Module):
    """
    Encodes a hyperedge's using 'convolutional' spatial approach
    """

    def __init__(self, feature_dim, embed_dim, hyperedge_aggregator, hypere_features, base_model=None):
        super(HyperedgeEncoder, self).__init__()

        self.feat_dim = feature_dim
        self.hyperedge_aggregator = hyperedge_aggregator
        if base_model != None:
            self.base_model = base_model
            #print("model", self.base_model)
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(
            torch.Tensor(embed_dim, 2*self.feat_dim))
        init.xavier_uniform_(self.weight)
        self.hypere_features = hypere_features #hyperedge representation at previous layer


    def compute(self, hyperedges):
        """
        Generates embeddings for a batch of hyperedges.
        nodes     -- list of nodes
        """
        #print('hypere encoding')
        #print('hyperedges size', len(hyperedges))
        agg_feats = self.hyperedge_aggregator.compute(hyperedges) #hyperedge representation at current layer
        #print("agg feat size", agg_feats.size())
        #print("agg feat", agg_feats[:5])
        if type(self.hypere_features) == nn.Embedding:
            self_feats = torch.Tensor(self.hypere_features(torch.LongTensor([int(i) for i in list(hyperedges)])))
        else:
            self_feats = torch.Tensor(self.hypere_features([int(i) for i in list(hyperedges)])).t()
        #print("self features size", self_feats.size())
        #print('agg feat size before cat', agg_feats.size())
        combined = torch.cat([self_feats, agg_feats], dim=1)
        #print('combined before embedding size', combined.size())
        combined = F.relu(self.weight.mm(combined.t()))
        #print('combined after embedding size', combined.t().size())
        #print('end hypere encoding')
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


MA = MeanAggregator(features_mat, hyp2nodes)
print(len(hyp2nodes))

hypere_features = nn.Embedding(len(hyp2nodes), 2)
hypere_features.weight = nn.Parameter(torch.zeros(len(hyp2nodes), 2))
enc = HyperedgeEncoder(2, 3, MA, hypere_features)
print(enccompute_aggregation(["0", "1"]))'''
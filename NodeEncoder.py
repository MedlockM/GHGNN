import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from HyperedgeAggregator import *
from NodeAggregator import *
from HyperedgeEncoder import *

class NodeEncoder(nn.Module):
    """
    Encodes a node's using 'convolutional' spatial approach
    """

    def __init__(self, feature_dim, embed_dim, node_aggregator, current_node_embeddings, base_model=None):
        super(NodeEncoder, self).__init__()

        self.feat_dim = feature_dim
        self.node_aggregator = node_aggregator
        if base_model != None:
            self.base_model = base_model
            #print("model", self.base_model)
        self.embed_dim = embed_dim
        self.current_node_embeddings = current_node_embeddings
        self.linear_transform = nn.Parameter(
            torch.FloatTensor(embed_dim, 2 * self.feat_dim))
        init.xavier_uniform_(self.linear_transform)


    def compute(self, node_ids):
        """
        Generates embeddings for a batch of nodes.

        Args:
            node_ids (torch.Tensor or list): A tensor or list of node IDs.

        Returns:
            torch.Tensor: The tensor containing embeddings for the batch of nodes.
        """
        # Convert node_ids to list if it is a tensor
        #node_ids = node_ids.tolist() if torch.is_tensor(node_ids) else node_ids
        print('nodes ids', node_ids)
        node_ids = torch.tensor(node_ids)
        # Aggregate features from neighboring nodes using the provided aggregator
        aggregated_nodes_embeddings = self.node_aggregator.compute(node_ids)

        # Retrieve embeddings for the nodes themselves, ensuring tensor type is consistent
        self_embeddings = self.current_node_embeddings(node_ids) if isinstance(self.current_node_embeddings,
                                                                               nn.Embedding) \
            else torch.tensor(self.current_node_embeddings(node_ids))

        # Concatenate self and aggregated embeddings to create combined feature vectors
        combined_features = torch.cat([self_embeddings, aggregated_nodes_embeddings], dim=1)

        # Apply the linear transformation and a non-linear activation function (ReLU)
        embeddings = F.relu(self.linear_transform.mm(combined_features.t()))

        # Return the resulting node embeddings as a transposed matrix to match input shape
        return embeddings.t()

#features_matrix = [[2, 4], [6, 9], [6, 6], [3, 3], [3, 3]]
#features_mat = nn.Embedding(5, 2)
#features_mat.weight = nn.Parameter(torch.FloatTensor(features_matrix), requires_grad=False)
#MA = MeanAggregator(features_mat, {"1" : ["2", "3", "4"], "2": ["1", "3"], "3": ["1", "2"], "4": ["1"], "5": []})

#nodes = ["3", "2"]

#enc = Encoder(2, 3, MA)

#enc.forward(nodes)

hyp2nodes = {}
hyp2nodes[0]=[0,1,2]
hyp2nodes[1]=[0,3]


nodes = [0, 1, 2, 3, 4]
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

print(node_enc.compute(nodes))


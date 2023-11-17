import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from RefNodeAggregator import *
from RefHyperedgeAggregator import *
from HyperedgeEncoder import *
class NodeEncoder(nn.Module):
    """
    Encodes node features by combining the node's own features with the aggregated features of its neighborhood.
    """

    def __init__(self, input_feature_dim, output_embedding_dim, aggregator, current_node_embeddings):
        """
        Initialize the NodeEncoder.

        Args:
            input_feature_dim (int): The dimensionality of the input features for each node.
            output_embedding_dim (int): The size of the output embedding for each node.
            aggregator (nn.Module): An instance of the node aggregator for aggregating features from neighboring nodes.
            current_node_embeddings (nn.Embedding or callable): Function or embedding layer to get current node embeddings.
        """
        super(NodeEncoder, self).__init__()

        # The dimensionality of input features and output embeddings
        self.input_feature_dim = input_feature_dim
        self.output_embedding_dim = output_embedding_dim

        # The aggregator and the current node embeddings, which are central to the encoding process
        self.aggregator = aggregator
        self.current_node_embeddings = current_node_embeddings

        # The linear transformation is a learned weight matrix that combines self and neighbor embeddings
        self.linear_transform = nn.Parameter(torch.FloatTensor(output_embedding_dim, 2 * input_feature_dim))
        nn.init.xavier_uniform_(self.linear_transform)  # Use Xavier initialization for the weight matrix

    def forward(self, node_ids):
        """
        Generates embeddings for a batch of nodes.

        Args:
            node_ids (torch.Tensor or list): A tensor or list of node IDs.

        Returns:
            torch.Tensor: The tensor containing embeddings for the batch of nodes.
        """
        # Convert node_ids to list if it is a tensor
        node_ids = node_ids.tolist() if torch.is_tensor(node_ids) else node_ids

        # Aggregate features from neighboring nodes using the provided aggregator
        aggregated_nodes_embeddings = self.aggregator.compute_aggregation(node_ids)

        # Retrieve embeddings for the nodes themselves, ensuring tensor type is consistent
        self_embeddings = self.current_node_embeddings(node_ids) if isinstance(self.current_node_embeddings,
                                                                               nn.Embedding) \
            else torch.tensor(self.current_node_embeddings(node_ids))

        # Concatenate self and neighbor embeddings to create combined feature vectors
        combined_features = torch.cat([self_embeddings, aggregated_nodes_embeddings], dim=1)

        # Apply the linear transformation and a non-linear activation function (ReLU)
        embeddings = F.relu(self.linear_transform.mm(combined_features.t()))

        # Return the resulting node embeddings as a transposed matrix to match input shape
        return embeddings.t()


class TestNodeEncoder(unittest.TestCase):
    def setUp(self):
        """
        Set up a simple testing environment with mocked parameters for the NodeEncoder.
        """
        # Specify the dimensions for the embeddings we will test with
        self.feature_dim = 4
        self.embedding_dim = 4

        def simple_hyperedge_embedding_lookup(hyperedge_ids):
            max_hyperedge_id = max(
                [hyperedge for hyperedges in self.node_to_hyperedges_valid.values() for hyperedge in hyperedges])
            embedding_dim = 3
            identity_matrix = torch.eye(max_hyperedge_id + 1, embedding_dim)
            print('hyperedges embeddings :', identity_matrix.data)
            return identity_matrix[hyperedge_ids]

        self.node_to_hyperedges_valid = {
            0: [0, 1],
            1: [0, 2],
            2: [2, 3]
        }
        # Initialize the mean Aggregator
        self.mean_aggregator = NodeMeanAggregator(simple_hyperedge_embedding_lookup, self.node_to_hyperedges_valid)

        # Define mock node embeddings as a function returning an identity matrix
        self.mock_node_embeddings = lambda indices: torch.eye(indices.max() + 1, )[indices]

        # Instantiate the NodeEncoder with the mocked parameters
        self.encoder = NodeEncoder(
            self.feature_dim, self.embedding_dim, self.mean_aggregator, self.mock_node_embeddings
        )


    def test_encoding(self):
        """
        Test the NodeEncoder forward pass for creating node embeddings.
        """
        # # Define node IDs as a test input
        # test_node_ids = torch.LongTensor([0, 1, 2, 3])
        #
        # # Encode node features and get the resulting embeddings
        # encoded_embeddings = self.encoder(test_node_ids)
        #
        # # Check that the shape of the encoded embeddings matches the expected output
        # expected_shape = (len(test_node_ids), self.embedding_dim)
        # self.assertEqual(encoded_embeddings.shape, expected_shape, "Encoded embeddings do not match expected shape.")

        node2hyperedges = {
            0: [0, 1],
            1: [0, 2],
            2: [2, 3]
        }
        nodes = [0, 1, 2]

        hyperedge2nodes = {}
        for j in nodes:
            hyperedge2nodes[j] = [i for i in node2hyperedges.keys() if j in node2hyperedges[i]]

        features_matrix = [[2, 4], [6, 9], [6, 6]]
        features_mat = nn.Embedding(3, 2)
        features_mat.weight = nn.Parameter(torch.FloatTensor(features_matrix), requires_grad=False)

        adj_dict = {"0": ["1", "2", "3"], "1": ["0", "2"], "2": ["0", "1"], "3": ["0"], "4": []}

        node_agg = NodeMeanAggregator(features_mat, node2hyperedges)

        hypere_features = nn.Embedding(len(node2hyperedges), 2)
        hypere_features.weight = nn.Parameter(torch.zeros(len(node2hyperedges), 2))

        hyperedge_enc = HyperedgeEncoder(2, 2, node_agg, hypere_features)

        hyperedge_agg = HyperedgeMeanAggregator(lambda hyperedge: hyperedge_enc.compute(hyperedge), hyperedge2nodes)

        node_enc = NodeEncoder(2, 5, hyperedge_agg, features_mat)

        print(node_enc.forward(nodes))

if __name__ == '__main__':
    unittest.main()
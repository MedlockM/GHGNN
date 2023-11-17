import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import unittest


class NodeMeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings by calculating the mean of the embeddings of its associated hyperedges.
    """

    def __init__(self, hyperedge_embedding_lookup, node_to_hyperedges):
        """
        Initializes the aggregator by defining the hyperedge embedding lookup function and storing the node-to-hyperedge mapping.

        :param hyperedge_embedding_lookup: A function or model component that takes a batch of hyperedge IDs as input and returns their embeddings.
        :param node_to_hyperedges: A dictionary where each key is a node ID and the value is a list of IDs of hyperedges containing that node.
        :raise ValueError: Raises an exception if the mapping does not follow the expected format.
        """
        super(NodeMeanAggregator, self).__init__()

        self.hyperedge_embedding_lookup = hyperedge_embedding_lookup
        self.node_to_hyperedges = node_to_hyperedges

    def compute_aggregation(self, nodes):
        """
        For each node in the provided list or tensor, computes the mean of its associated hyperedge embeddings.

        :param nodes: tensor or list containing node IDs
        :return: tensor containing the aggregated embeddings for each node
        """
        # Convert nodes to a list if they are in a tensor format
        nodes = nodes.tolist() if isinstance(nodes, torch.Tensor) else nodes

        # Retrieve the list of associated hyperedges for each node
        print('nodes to hyp', self.node_to_hyperedges)
        associated_hyperedges_lists = [
            self.node_to_hyperedges[node_id] for node_id in nodes
        ]

        # Flatten the list of hyperedges and remove duplicates to identify unique hyperedges
        unique_hyperedges = list(set(hyperedge for hyperedges in associated_hyperedges_lists for hyperedge in hyperedges))

        # Create a mapping of each unique hyperedge to an index position
        hyperedge_to_index = {hyperedge_id: index for index, hyperedge_id in enumerate(unique_hyperedges)}

        # Initialize a mask to zero - it will be of size [number of nodes, number of unique hyperedges]
        node_hyperedge_mask = torch.zeros(len(nodes), len(unique_hyperedges))

        # Fill the mask with ones where the node is associated with the hyperedge
        hyperedge_indices = [hyperedge_to_index[hyperedge] for hyperedges in associated_hyperedges_lists for hyperedge in hyperedges]
        node_indices = [i for i, hyperedges in enumerate(associated_hyperedges_lists) for _ in hyperedges]
        node_hyperedge_mask[node_indices, hyperedge_indices] = 1

        # Compute the mean aggregation by normalizing the mask and multiplying it with the hyperedge embeddings
        hyperedge_count = node_hyperedge_mask.sum(1, keepdim=True)
        normalized_mask = node_hyperedge_mask / hyperedge_count

        # Obtain the hyperedge embeddings using the provided embedding function
        embedding_matrix = self.hyperedge_embedding_lookup(torch.tensor(unique_hyperedges))

        # Apply the mean aggregation operation to get the aggregated node embeddings
        aggregated_node_embeddings = normalized_mask.mm(embedding_matrix)

        return aggregated_node_embeddings

class NodeMaxAggregator(nn.Module):
    """
    Aggregates a node's embeddings by calculating the mean of the embeddings of its associated hyperedges.
    """

    def __init__(self, hyperedge_embedding_lookup, node_to_hyperedges):
        """
        Initializes the aggregator by defining the hyperedge embedding lookup function and storing the node-to-hyperedge mapping.

        :param hyperedge_embedding_lookup: A function or model component that takes a batch of hyperedge IDs as input and returns their embeddings.
        :param node_to_hyperedges: A dictionary where each key is a node ID and the value is a list of IDs of hyperedges containing that node.
        :raise ValueError: Raises an exception if the mapping does not follow the expected format.
        """
        super(NodeMaxAggregator, self).__init__()

        self.hyperedge_embedding_lookup = hyperedge_embedding_lookup
        self.node_to_hyperedges = node_to_hyperedges

    def compute_aggregation(self, nodes):
        """
        For each node in the provided list or tensor, computes the max pooling of its associated hyperedge embeddings.

        :param nodes: tensor or list containing node IDs
        :return: tensor containing the aggregated embeddings for each node
        """
        # Convert nodes to a list if they are in a tensor format
        nodes = nodes.tolist() if isinstance(nodes, torch.Tensor) else nodes

        # Initialize a tensor to store the maximum aggregated embeddings
        max_aggregated_embeddings = None

        # Process each node to find the maximum hyperedge embedding
        for node_id in nodes:
            # Retrieve associated hyperedge embeddings
            hyperedge_ids = self.node_to_hyperedges[node_id]
            hyperedge_embeddings = self.hyperedge_embedding_lookup(torch.tensor(hyperedge_ids, dtype=torch.long))

            # Calculate the max embedding among the associated hyperedges
            max_embedding, _ = torch.max(hyperedge_embeddings, dim=0, keepdim=True)

            # Accumulate the results
            if max_aggregated_embeddings is None:
                max_aggregated_embeddings = max_embedding
            else:
                max_aggregated_embeddings = torch.cat((max_aggregated_embeddings, max_embedding), dim=0)

        return max_aggregated_embeddings
# Tests
class TestNodeAggregator(unittest.TestCase):
    def setUp(self):
        """
        Setup for NodeMeanAggregator testing with validated input formats.
        """
        def simple_hyperedge_embedding_lookup(hyperedge_ids):
            max_hyperedge_id = max(
                [hyperedge for hyperedges in self.node_to_hyperedges_valid.values() for hyperedge in hyperedges])
            embedding_dim = 3
            identity_matrix = torch.eye(max_hyperedge_id + 1, embedding_dim)
            print(identity_matrix.data)
            return identity_matrix[hyperedge_ids]

        self.node_to_hyperedges_valid = {
            0: [0, 1],
            1: [0, 2],
            2: [2, 3]
        }
        # Initialize the mean Aggregator
        self.mean_aggregator = NodeMeanAggregator(simple_hyperedge_embedding_lookup, self.node_to_hyperedges_valid)
        self.max_aggregator = NodeMaxAggregator(simple_hyperedge_embedding_lookup, self.node_to_hyperedges_valid)
    def test_mean_aggregate_embeddings_single_node(self):
        """
        Test the compute_aggregation method with valid inputs.
        """

        # Test using a single node ID
        node_ids_single = torch.tensor([0])
        aggregated_embeddings_single = self.mean_aggregator.compute_aggregation(node_ids_single)
        expected_output_single = torch.tensor([[0.5, 0.5, 0.0]])  # Mean of embeddings for hyperedges 0 and 1
        torch.testing.assert_close(aggregated_embeddings_single, expected_output_single)
    def test_mean_aggregate_embeddings_multiple_nodes(self):
        # Test using multiple node IDs
        node_ids_multiple = torch.tensor([0, 1, 2])
        aggregated_embeddings_multiple = self.mean_aggregator.compute_aggregation(node_ids_multiple)
        print('agg :', aggregated_embeddings_multiple.data)
        expected_output_multiple = torch.tensor([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 0.5]
        ])  # Mean of embeddings for corresponding hyperedges of each node
        torch.testing.assert_close(aggregated_embeddings_multiple, expected_output_multiple)

    def test_max_aggregate_embeddings(self):
        # Test using multiple node IDs
        node_ids = torch.tensor([0, 1, 2])
        aggregated_embeddings = self.max_aggregator.compute_aggregation(node_ids)

        # Define the expected output based on the max operation
        expected_output = torch.tensor([
            [1.0, 1.0, 0.0],  # Max of embeddings for hyperedges 0 and 1
            [1.0, 0.0, 1.0],  # Max of embeddings for hyperedges 0 and 2
            [0.0, 0.0, 1.0]  # Max of embeddings for hyperedges 2 and 3
        ], dtype=torch.float32)  # Explicitly specify dtype as torch.float32

        # Test the aggregated embeddings against the expected results
        torch.testing.assert_close(aggregated_embeddings, expected_output)

class NodeAttAggregator(nn.Module):
    """
    Implementation of a Graph Attention Aggregator for nodes in a hypergraph,
    using an attention mechanism to weigh the importance of hyperedges.
    """

    def __init__(self, features_calc, hyperedges_to_nodes_mapping, input_dim, output_dim, dropout=0.6, alpha=0.2, concat=True):
        super(NodeAttAggregator, self).__init__()
        self.features_calc = features_calc  # Function to compute node embeddings
        self.hyperedges_to_nodes_mapping = hyperedges_to_nodes_mapping  # Mapping from hyperedges to nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha  # Negative input slope for LeakyReLU
        self.concat = concat

        # Initialize weights using Xavier Initialization
        self.query_weight = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.query_weight.data, gain=1.414)
        self.key_weight = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.key_weight.data, gain=1.414)
        self.value_weight = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.value_weight.data, gain=1.414)

        # Initialize the attention coefficient vector
        self.attention_coef = nn.Parameter(torch.zeros(size=(input_dim, 1)))
        nn.init.xavier_uniform_(self.attention_coef.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def compute_attention(self, hyperedges):
        hyperedges = hyperedges.tolist() if torch.is_tensor(hyperedges) else hyperedges
        if isinstance(self.features_calc, nn.Embedding):
            node_embeddings = self.features_calc(torch.tensor(hyperedges, dtype=torch.long))
        else:
            node_embeddings = self.features_calc(hyperedges)

        # Calculate node and hyperedge embeddings with query and key weights
        hyperedge_query = torch.mm(node_embeddings, self.query_weight)
        node_key = torch.mm(node_embeddings, self.key_weight)

        # Compute attention scores using the dot product, and normalize them
        attention_scores = torch.mm(hyperedge_query, node_key.T) / math.sqrt(self.output_dim)
        attention_mask = self._get_attention_mask(hyperedges)
        masked_attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_probs = F.softmax(masked_attention_scores, dim=1)

        # Optionally apply dropout to attention probabilities before summing
        # attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)

        # Aggregate nodes using the attention mechanism
        attention_output = torch.matmul(attention_probs, node_key)

        return attention_output

    def _get_attention_mask(self, hyperedges):
        # Crée une liste unique des nœuds impliqués en explorant toutes les listes de nœuds associées aux hyperarêtes fournies.
        unique_nodes = list(
            set([node for hyperedge in hyperedges for node in self.hyperedges_to_nodes_mapping[hyperedge]]))

        # Établit une correspondance entre chaque nœud unique et un indice, qui sera utilisé pour indexer le masque d'attention.
        unique_nodes_indices = {node: idx for idx, node in enumerate(unique_nodes)}

        # Crée un tenseur rempli de '-inf', ce qui représente un score très bas pour l'attention (et deviendra zéro après softmax).
        # La taille du masque est [nombre d'hyperarêtes, nombre de nœuds uniques].
        mask = torch.full((len(hyperedges), len(unique_nodes)), fill_value=float('-inf'))

        # Remplit le masque avec des valeurs de zéro aux positions où l'attention doit être appliquée.
        # Pour chaque hyperarête et chaque nœud correspondant, on définit le masque à zéro.
        for i, hyperedge in enumerate(hyperedges):
            for node in self.hyperedges_to_nodes_mapping[hyperedge]:
                mask[i, unique_nodes_indices[node]] = 0  # Utilise le score d'attention réel plutôt que '-inf'.

        # Retourne le masque d'attention qui sera utilisé dans le calcul d'attention.
        return mask

# Tests for NodeAttAggregator (construction and usage) would go here

# Running the test when the script is executed
if __name__ == '__main__':
    unittest.main()

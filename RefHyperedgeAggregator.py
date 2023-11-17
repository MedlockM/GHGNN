import torch
import torch.nn as nn
import unittest


class HyperedgeMeanAggregator(nn.Module):
    """A module for aggregating embeddings of hyperedges' internal nodes."""

    def __init__(self, node_embedding_lookup, hyperedge_to_nodes):
        """
        Initialize the HyperedgeAggregator.

        Args:
            node_embedding_lookup (callable):
                A function that takes node indices as input and returns their embeddings.
            hyperedge_to_nodes (dict):
                A dictionary mapping hyperedges to lists of internal nodes.
        """
        super(HyperedgeMeanAggregator, self).__init__()
        self.node_embedding_lookup = node_embedding_lookup
        self.hyperedge_to_nodes = hyperedge_to_nodes

    def compute_aggregation(self, hyperedges):
        """
        Aggregate hyperedges embeddings using the mean of their internal nodes' embeddings.

        Args:
            hyperedges (list or torch.Tensor):
                List or tensor of hyperedge IDs.

        Returns:
            torch.Tensor: Tensor containing the mean aggregated embeddings for each hyperedge.
        """
        # Convert hyperedges to list if input is tensor
        hyperedges = hyperedges.tolist() if torch.is_tensor(hyperedges) else hyperedges
        # Create mask for mean aggregation and retrieve unique internal nodes
        mask, unique_node_ids = self._create_mask(hyperedges)
        # Get embeddings for unique internal nodes
        unique_node_ids = [int(node_id) for node_id in unique_node_ids]
        embedding_matrix = self.node_embedding_lookup(torch.tensor(unique_node_ids, dtype=torch.long))
        # Return the mean aggregated embeddings using matrix multiplication
        return mask.mm(embedding_matrix)

    def _create_mask(self, hyperedges):
        """
        Create a mask for mean aggregation calculation.

        Args:
            hyperedges (list): List of hyperedge IDs.

        Returns:
            torch.Tensor: Mask for the mean aggregation.
            list: List of unique internal node IDs.
        """
        # Flatten list of internal nodes for all hyperedges and get unique nodes
        internal_node_lists = [self.hyperedge_to_nodes[h] for h in hyperedges]
        unique_node_ids = list(set(node for sublist in internal_node_lists for node in sublist))
        # Create mask to compute mean aggregation
        mask = torch.zeros(len(hyperedges), len(unique_node_ids))
        # Map internal node IDs to their indices in unique_node_ids
        unique_node_indices = {node: idx for idx, node in enumerate(unique_node_ids)}
        # Fill mask with ones based on hyperedge-to-internal nodes mapping
        for hyperedge_idx, sublist in enumerate(internal_node_lists):
            for node_id in sublist:
                mask[hyperedge_idx, unique_node_indices[node_id]] = 1
        # Normalize rows of mask to sum to 1
        num_nodes = mask.sum(1, keepdim=True)
        normalized_mask = mask.div(num_nodes)
        return normalized_mask, unique_node_ids

class HyperedgeMaxAggregator(nn.Module):
    """A module for aggregating embeddings of hyperedges' internal nodes."""

    def __init__(self, node_embedding_lookup, hyperedge_to_nodes):
        """
        Initialize the HyperedgeAggregator.

        Args:
            node_embedding_lookup (callable):
                A function that takes node indices as input and returns their embeddings.
            hyperedge_to_nodes (dict):
                A dictionary mapping hyperedges to lists of internal nodes.
        """
        super(HyperedgeMaxAggregator, self).__init__()
        self.node_embedding_lookup = node_embedding_lookup
        self.hyperedge_to_nodes = hyperedge_to_nodes
    def max_aggregation(self, hyperedges):
        """
        Aggregate hyperedges embeddings using the max of their internal nodes' embeddings.

        Args:
            hyperedges (list or torch.Tensor):
                List or tensor of hyperedge IDs.

        Returns:
            torch.Tensor: Tensor containing the max aggregated embeddings for each hyperedge.
        """
        # Convert hyperedges to list if input is tensor
        hyperedges = hyperedges.tolist() if torch.is_tensor(hyperedges) else hyperedges
        # List to store max aggregated embeddings for each hyperedge
        max_aggregated_embeddings = []
        # Retrieve and max-aggregate the embeddings for each hyperedge's internal nodes
        for hyperedge in hyperedges:
            internal_node_ids = [int(node) for node in self.hyperedge_to_nodes[hyperedge]]
            internal_node_embeddings = self.node_embedding_lookup(torch.tensor(internal_node_ids, dtype=torch.long))
            max_embedding, _ = torch.max(internal_node_embeddings, dim=0, keepdim=True)
            max_aggregated_embeddings.append(max_embedding)
        # Return the max aggregated embeddings for all hyperedges
        return torch.cat(max_aggregated_embeddings, dim=0)

    def _create_mask(self, hyperedges):
        """
        Create a mask for mean aggregation calculation.

        Args:
            hyperedges (list): List of hyperedge IDs.

        Returns:
            torch.Tensor: Mask for the mean aggregation.
            list: List of unique internal node IDs.
        """
        # Flatten list of internal nodes for all hyperedges and get unique nodes
        internal_node_lists = [self.hyperedge_to_nodes[h] for h in hyperedges]
        unique_node_ids = list(set(node for sublist in internal_node_lists for node in sublist))
        # Create mask to compute mean aggregation
        mask = torch.zeros(len(hyperedges), len(unique_node_ids))
        # Map internal node IDs to their indices in unique_node_ids
        unique_node_indices = {node: idx for idx, node in enumerate(unique_node_ids)}
        # Fill mask with ones based on hyperedge-to-internal nodes mapping
        for hyperedge_idx, sublist in enumerate(internal_node_lists):
            for node_id in sublist:
                mask[hyperedge_idx, unique_node_indices[node_id]] = 1
        # Normalize rows of mask to sum to 1
        num_nodes = mask.sum(1, keepdim=True)
        normalized_mask = mask.div(num_nodes)
        return normalized_mask, unique_node_ids


# Test class for the HyperedgeAggregator
class TestHyperedgeAggregator(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with a dummy node embedding lookup and a hyperedge to nodes mapping."""
        embedding_dim = 4
        # Define dummy embeddings as identity matrix for simplification
        self.embeddings = torch.eye(5, embedding_dim)
        # Define node embedding lookup function
        self.node_embedding_lookup = lambda indices: self.embeddings[indices]
        # Define hyperedge to nodes mapping for tests
        self.hyperedge_to_nodes = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4]}
        # Create instance of HyperedgeAggregator for testing
        self.mean_aggregator = HyperedgeMeanAggregator(self.node_embedding_lookup, self.hyperedge_to_nodes)
        self.max_aggregator = HyperedgeMaxAggregator(self.node_embedding_lookup, self.hyperedge_to_nodes)

    def test_mean_aggregation(self):
        """Test the compute_aggregation method of the HyperedgeAggregator."""
        hyperedges = torch.tensor([0, 1, 2, 3])
        expected_mean_embeddings = torch.tensor([
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.5]
        ])
        aggregated_embeddings = self.mean_aggregator.compute_aggregation(hyperedges)
        self.assertTrue(torch.equal(aggregated_embeddings, expected_mean_embeddings))

    def test_max_aggregation(self):
        """Test the max_aggregation method of the HyperedgeAggregator."""
        hyperedges = torch.tensor([0, 1, 2, 3])
        expected_max_embeddings = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        aggregated_embeddings = self.max_aggregator.max_aggregation(hyperedges)
        self.assertTrue(torch.equal(aggregated_embeddings, expected_max_embeddings))


if __name__ == '__main__':
    unittest.main()